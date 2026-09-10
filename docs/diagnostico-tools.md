# Diagnóstico de herramientas rechazadas

Desde el cambio del 7 de septiembre de 2026, el worker real captura errores de
parseo y validación de llamadas nativas. No cambia qué argumentos se aceptan,
no repara llamadas y no hace reintentos. Una llamada rechazada sigue sin
ejecutarse y la respuesta fallida no genera un snapshot válido.

## Activación y privacidad

El worker real guarda las capturas por defecto en `state/tool-errors/`, relativo
al directorio de trabajo (la raíz de Qwasar con el servicio instalado). El worker
de pruebas `--fake` no escribe capturas por defecto. Un proceso ya iniciado
necesita reiniciarse para cargar este cambio: editar los archivos no activa la
captura en el worker residente.

`QWASAR_TOOL_DIAGNOSTICS_DIR` permite elegir otro directorio; una cadena vacía
desactiva la captura. No se cambia automáticamente la configuración de systemd.

- Directorio privado `0700`, archivos `0600`, escritura mediante archivo temporal
  y reemplazo atómico. Se rechaza un directorio final que sea un enlace simbólico.
- Se conservan como máximo las últimas 20 capturas, hasta 1 MiB cada una. La
  retención se aplica al guardar, no por antigüedad ni mediante un servicio aparte.
- Si la evidencia excede el límite, se guarda solo un aviso con tamaño y hash;
  ese registro no sirve para reconstruir la llamada. No se trunca silenciosamente.
- Se captura el bloque XML de herramientas exactamente como lo recibió el parser
  (texto decodificado, no IDs de tokens), los esquemas de herramientas, argumentos
  interpretados hasta el fallo, herramienta e índice de llamada, etapa, error,
  ruta de validación, tipos esperado/recibido, ID de respuesta, identidad del
  runtime y hash del código del parser.
- No se copia la conversación ni el razonamiento. **Los argumentos y los esquemas
  pueden contener secretos o código privado.** No subir estos archivos a Git,
  publicarlos ni compartirlos sin revisar. `state/` ya está ignorado por Git.
- El protocolo del error solo añade `diagnostic_id` y `diagnostic_status`
  (`captured`, `size_limit` o `unavailable`). No añade XML, valores ni rutas locales.
  Si falla el almacenamiento, se mantiene el error original y el worker puede
  seguir atendiendo solicitudes después de su reset habitual.

## Cuando vuelva a ocurrir

Buscar el JSON correspondiente al `diagnostic_id` del error SSE o del error
almacenado en `response.error` de la base de datos. Pi puede mostrar solamente el
mensaje original; no necesariamente presenta los campos adicionales.

Desde la raíz del repositorio:

```bash
PYTHONPATH=src .venv/bin/python -m qwasar_runtime.diagnostics state/tool-errors/tool-error-ID.json
```

Reemplazar `ID` por el identificador real. Este comando no carga el modelo, no
ejecuta herramientas y no usa la GPU. Compara el bloque capturado con su
interpretación por el parser actual y el resultado de validación. El informe
omite los valores de argumentos. Si el hash del parser cambió, lo indica; una
reproducción con el parser actual no demuestra qué hacía otra versión.

La comparación es una reproducción de nuestro parser, **no un validador
independiente ni una prueba de que el modelo sea culpable**. Revisar el XML y el
esquema privados permite distinguir si el modelo emitió un tipo incorrecto o si
Qwasar lo interpretó mal. Convertir la evidencia real en una regresión sanitizada
antes de corregirlo. En errores de esquemas compuestos, la ruta puede corresponder
al nodo `anyOf`/`oneOf`/`allOf` que no se satisface, no a cada rama interna.
