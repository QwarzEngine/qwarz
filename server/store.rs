use crate::api;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::{path::Path, sync::Mutex};

pub struct Store {
    connection: Mutex<Connection>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, String> {
        let connection = Connection::open(path).map_err(|error| error.to_string())?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS responses(id TEXT PRIMARY KEY, status TEXT NOT NULL, request TEXT NOT NULL,
            result TEXT NOT NULL, snapshot TEXT, history_key TEXT, idempotency TEXT UNIQUE);
            CREATE INDEX IF NOT EXISTS response_history ON responses(history_key);").map_err(|error|error.to_string())?;
        let mut statement = connection
            .prepare("SELECT id FROM responses WHERE status='in_progress'")
            .map_err(|error| error.to_string())?;
        let stale: Vec<String> = statement
            .query_map([], |row| row.get(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())?;
        drop(statement);
        for id in stale {
            let failure = json!({"id":id,"object":"response","status":"failed","error":{"code":"server_restarted","message":"server stopped before committing response"},"output":[]});
            connection
                .execute(
                    "UPDATE responses SET status='failed',result=?1,snapshot=NULL WHERE id=?2",
                    params![failure.to_string(), id],
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    pub fn begin(&self, id: &str, request: &Value, key: Option<&str>) -> Result<(), String> {
        let result = json!({"id":id,"object":"response","status":"in_progress","model":api::MODEL,"output":[]});
        self.connection.lock().unwrap().execute("INSERT INTO responses(id,status,request,result,idempotency) VALUES(?1,'in_progress',?2,?3,?4)", params![id,request.to_string(),result.to_string(),key]).map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn finish(
        &self,
        id: &str,
        status: &str,
        result: &Value,
        snapshot: Option<&Value>,
    ) -> Result<(), String> {
        let snapshot = if status == "completed" {
            snapshot
        } else {
            None
        };
        if status == "completed" && snapshot.is_none() {
            return Err("completed response requires snapshot".into());
        }
        let history_key = snapshot
            .and_then(|value| value["messages"].as_array())
            .map(|messages| api::history_key(messages));
        self.connection.lock().unwrap().execute("UPDATE responses SET status=?1,result=?2,snapshot=?3,history_key=?4 WHERE id=?5 AND status='in_progress'",params![status,result.to_string(),snapshot.map(Value::to_string),history_key,id]).map_err(|error|error.to_string()).and_then(|count| if count == 1 {Ok(())} else {Err("response was already finalized or does not exist".into())})
    }

    pub fn get(&self, id: &str) -> Result<Value, String> {
        let text: String = self
            .connection
            .lock()
            .unwrap()
            .query_row("SELECT result FROM responses WHERE id=?1", [id], |row| {
                row.get(0)
            })
            .map_err(|error| error.to_string())?;
        serde_json::from_str(&text).map_err(|error| error.to_string())
    }

    pub fn parent(&self, id: &str) -> Result<Value, String> {
        let text: String = self.connection.lock().unwrap().query_row("SELECT snapshot FROM responses WHERE id=?1 AND status='completed' AND snapshot IS NOT NULL",[id], |row|row.get(0)).map_err(|_|"parent is unknown or not completed".to_string())?;
        serde_json::from_str(&text).map_err(|error| error.to_string())
    }

    pub fn match_parent(&self, messages: &[Value]) -> Result<Option<Value>, String> {
        let connection = self.connection.lock().unwrap();
        for index in (0..messages.len()).rev() {
            if messages[index]["role"] != "assistant" {
                continue;
            }
            let key = api::history_key(&messages[..=index]);
            let mut statement = connection.prepare("SELECT snapshot FROM responses WHERE history_key=?1 AND status='completed' ORDER BY rowid DESC")
                .map_err(|error| error.to_string())?;
            let rows = statement
                .query_map([key], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let supplied: Vec<&Value> = messages[..=index]
                .iter()
                .filter(|message| message["role"] != "system")
                .collect();
            let mut matched: Option<Value> = None;
            for row in rows {
                let snapshot: Value =
                    serde_json::from_str(&row.map_err(|error| error.to_string())?)
                        .map_err(|error| error.to_string())?;
                let known: Vec<&Value> = snapshot["messages"]
                    .as_array()
                    .ok_or("invalid snapshot messages")?
                    .iter()
                    .filter(|message| message["role"] != "system")
                    .collect();
                if supplied.len() != known.len()
                    || supplied.iter().zip(known.iter()).any(|(incoming, stored)| {
                        incoming
                            .get("reasoning_content")
                            .is_some_and(|thought| thought != &stored["reasoning_content"])
                    })
                {
                    continue;
                }
                if let Some(previous) = &matched {
                    if previous["tape"] != snapshot["tape"]
                        || previous["segments"] != snapshot["segments"]
                    {
                        return Err(
                            "ambiguous history: supply preserved reasoning or previous_response_id"
                                .into(),
                        );
                    }
                } else {
                    matched = Some(snapshot);
                }
            }
            if matched.is_some() {
                return Ok(matched);
            }
        }
        Ok(None)
    }

    pub fn idempotent(&self, key: &str) -> Result<Option<(String, Value, Value)>, String> {
        let row: Option<(String, String, String)> = self
            .connection
            .lock()
            .unwrap()
            .query_row(
                "SELECT id,request,result FROM responses WHERE idempotency=?1",
                [key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        row.map(|(id, request, result)| {
            Ok((
                id,
                serde_json::from_str(&request).map_err(|error| error.to_string())?,
                serde_json::from_str(&result).map_err(|error| error.to_string())?,
            ))
        })
        .transpose()
    }
}
