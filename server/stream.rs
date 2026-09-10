use serde_json::{Value, json};

pub struct ResponsesStream {
    id: String,
    sequence: u64,
    items: Vec<String>,
}

impl ResponsesStream {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            sequence: 0,
            items: Vec::new(),
        }
    }

    fn event(&mut self, mut value: Value) -> Value {
        value["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        value
    }

    pub fn created(&mut self) -> Value {
        self.event(json!({"type":"response.created","response":{"id":self.id,"object":"response","status":"in_progress","model":crate::api::MODEL,"output":[]}}))
    }

    fn add(&mut self, item: &Value, events: &mut Vec<Value>) -> (usize, bool) {
        let id = item["id"].as_str().unwrap();
        if let Some(index) = self.items.iter().position(|known| known == id) {
            return (index, false);
        }
        let index = self.items.len();
        self.items.push(id.into());
        let mut initial = item.clone();
        initial["status"] = json!("in_progress");
        match item["type"].as_str() {
            Some("message") => initial["content"] = json!([]),
            Some("reasoning") => initial["summary"] = json!([]),
            Some("function_call") => initial["arguments"] = json!(""),
            _ => {}
        }
        events.push(self.event(
            json!({"type":"response.output_item.added","output_index":index,"item":initial}),
        ));
        match item["type"].as_str() {
            Some("message") => events.push(self.event(json!({"type":"response.content_part.added","item_id":id,"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}))),
            Some("reasoning") => events.push(self.event(json!({"type":"response.reasoning_summary_part.added","item_id":id,"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}}))),
            _ => {}
        }
        (index, true)
    }

    pub fn delta(&mut self, reasoning: bool, text: &Value) -> Vec<Value> {
        let item = if reasoning {
            json!({"id":format!("rs_{}",self.id),"type":"reasoning","summary":[]})
        } else {
            json!({"id":format!("msg_{}",self.id),"type":"message","role":"assistant","content":[]})
        };
        let mut events = Vec::new();
        let (index, _) = self.add(&item, &mut events);
        let mut event = json!({"type":if reasoning {"response.reasoning_summary_text.delta"} else {"response.output_text.delta"},"item_id":item["id"],"output_index":index,"delta":text});
        event[if reasoning {
            "summary_index"
        } else {
            "content_index"
        }] = json!(0);
        events.push(self.event(event));
        events
    }

    pub fn ordered_response(&self, mut response: Value) -> Value {
        if let Some(output) = response["output"].as_array_mut() {
            output.sort_by_key(|item| {
                self.items
                    .iter()
                    .position(|id| item["id"] == *id)
                    .unwrap_or(usize::MAX)
            });
        }
        response
    }

    pub fn finish(&mut self, response: &Value) -> Vec<Value> {
        let mut events = Vec::new();
        for item in response["output"].as_array().unwrap() {
            let (index, added) = self.add(item, &mut events);
            let id = &item["id"];
            match item["type"].as_str() {
                Some("message") => {
                    let part = &item["content"][0];
                    if added {
                        events.push(self.event(json!({"type":"response.output_text.delta","item_id":id,"output_index":index,"content_index":0,"delta":part["text"]})));
                    }
                    events.push(self.event(json!({"type":"response.output_text.done","item_id":id,"output_index":index,"content_index":0,"text":part["text"]})));
                    events.push(self.event(json!({"type":"response.content_part.done","item_id":id,"output_index":index,"content_index":0,"part":part})));
                }
                Some("reasoning") => {
                    let part = &item["summary"][0];
                    if added {
                        events.push(self.event(json!({"type":"response.reasoning_summary_text.delta","item_id":id,"output_index":index,"summary_index":0,"delta":part["text"]})));
                    }
                    events.push(self.event(json!({"type":"response.reasoning_summary_text.done","item_id":id,"output_index":index,"summary_index":0,"text":part["text"]})));
                    events.push(self.event(json!({"type":"response.reasoning_summary_part.done","item_id":id,"output_index":index,"summary_index":0,"part":part})));
                }
                Some("function_call") => {
                    events.push(self.event(json!({"type":"response.function_call_arguments.delta","item_id":id,"output_index":index,"delta":item["arguments"]})));
                    events.push(self.event(json!({"type":"response.function_call_arguments.done","item_id":id,"output_index":index,"arguments":item["arguments"]})));
                }
                _ => {}
            }
            events.push(self.event(
                json!({"type":"response.output_item.done","output_index":index,"item":item}),
            ));
        }
        events.push(self.event(json!({"type":format!("response.{}",response["status"].as_str().unwrap_or("failed")),"response":response})));
        events
    }
}
