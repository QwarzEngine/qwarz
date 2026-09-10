# Arranque automático de Qwasar

Unidad: [`integrations/systemd/qwasar.service`](../integrations/systemd/qwasar.service).
Instalación local: `~/.config/systemd/user/qwasar.service`, enlazada al archivo
del repositorio. Es un servicio **de usuario systemd**, no un proceso ejecutado
como root ni una aplicación de inicio del escritorio.

## Qué arranca

- RTX 5090, GPU 0; `CUDA_VISIBLE_DEVICES=0`.
- `/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw`.
- EXL3 5 bpw nominal, cabeza a 6 bits, MTP a 4 bits.
- MTP con 4 tokens draft y caché K8/V4, fijados por el backend v1.
- Ventana nativa de 262.144 posiciones, fijada por el servicio v1.
- Perfil Flash de prefill, bloques de 8.192 tokens.
- Endpoint local `http://127.0.0.1:8800/v1`, modelo `qwasar-qwen38-27b`.
- La misma base durable `state/qwasar.db`; las sesiones no se borran al parar.

El arranque comprueba la ocupación de GPU 0. No mata otros procesos para hacer
sitio. El worker verifica el hash completo del artefacto antes de cargar CUDA.
El servicio permanece en `activating` hasta que su propio worker publica
`ready`; no basta con que otro proceso responda HTTP 200 en el puerto.

## Arranque sin iniciar sesión

El usuario `rekeyea` ya tenía **`Linger=yes`**. Con la unidad habilitada en
`default.target`, el gestor de usuario puede iniciar Qwasar al arrancar el
sistema, sin esperar al login gráfico, y mantenerlo al cerrar sesión.

```bash
loginctl show-user rekeyea -p Linger
systemctl --user is-enabled qwasar.service
```

El servicio antiguo `qwen38-27b-exl3.service` se deshabilitó del autoarranque
para que no compita por la 5090. Su archivo y el repositorio donante se conservan.
`sglang-lfm.service`, en la RTX 3090 Ti, permanece activo y habilitado.

Esto no implica restauración instantánea de la caché GPU: después del arranque
se cargan los pesos y un historial recuperado puede necesitar prefill frío.
Un home que no esté disponible antes del login impediría este arranque temprano.

## Uso diario

```bash
systemctl --user status qwasar.service
systemctl --user restart qwasar.service
systemctl --user stop qwasar.service
systemctl --user start qwasar.service
journalctl --user -u qwasar.service -n 100 -f
```

También funcionan los comandos existentes:

```bash
python3 scripts/qwasar.py status
python3 scripts/qwasar.py stop
python3 scripts/qwasar.py start
python3 scripts/qwasar.py logs
```

Cuando detectan esta unidad instalada, delegan en systemd: no crean otro
daemon con `nohup`. Pi sigue usando `scripts/pi_qwasar.sh` desde el proyecto en
el que quieras trabajar. Los logs nuevos van al journal; `state/server.log`
conserva los logs históricos del arranque manual anterior.

Para desactivar también el autoarranque:

```bash
systemctl --user disable --now qwasar.service
```

Un `stop` manual no provoca relanzamiento. Ante una salida anormal, systemd
reintenta tras 15 segundos. Durante la parada envía SIGTERM al supervisor y
permite la terminación del worker; a los 45 segundos puede eliminar el resto
del grupo de procesos de Qwasar. No afecta a procesos de otras unidades.

## Instalación y actualizaciones

La unidad está preparada para la disposición actual bajo `~/Documents/llm` y
`~/models`; no es una plantilla portable a cualquier directorio sin editarla.
El entorno declara rutas de Python/CUDA y no depende de iniciar un shell
interactivo, mise o el escritorio. No descarga pesos ni compila Rust al arrancar.

Para instalar desde un arranque manual anterior, estando el motor sin trabajo:

```bash
cd /home/rekeyea/Documents/llm/qwasar
cargo build --release --locked --offline
python3 scripts/qwasar.py stop
systemctl --user disable qwen38-27b-exl3.service
systemctl --user enable "$PWD/integrations/systemd/qwasar.service"
systemctl --user daemon-reload
systemctl --user start qwasar.service
```

En otro usuario sin linger sería necesario habilitarlo con
`loginctl enable-linger USUARIO`, sujeto a los permisos de la máquina. Aquí no
fue necesario cambiarlo. Consultar `man loginctl`, `man systemd.service` y
`man systemd.kill` para las reglas del gestor instalado.

Después de cambiar código Rust: compilar release y reiniciar. Después de cambiar
la unidad: ejecutar `systemd-analyze --user verify` sobre el archivo,
`systemctl --user daemon-reload` y reiniciar. Un cambio del perfil requiere
editar explícitamente `--prefill flash` en la unidad: el launcher no modifica
una configuración persistente al recibir `start --prefill baseline`.

No reiniciar mientras Pi esté generando salvo que se quiera interrumpir ese
trabajo. La habilitación para boot se comprueba sin reiniciar toda la máquina.

## Verificación de la instalación local

El 2026-09-06 se verificaron la sintaxis con `systemd-analyze --user verify`,
290 tests Python, la primera carga bajo systemd y un reinicio gestionado. El
worker anterior desapareció y el nuevo respondió `BOOT_READY` con el perfil
5 bpw + MTP + K8/V4, Flash y 262.144 posiciones. La unidad quedó `enabled` y
`active/running`, con `Linger=yes`; SGLang en GPU 1 conservó su proceso.
No se reinició la máquina ni se simuló un fallo fatal del supervisor.

Evidencia local: `results/systemd-install/before.json` y
`results/systemd-install/verified.json`. La recuperación ante una salida anormal
queda configurada con `Restart=on-failure`; la prueba realizada fue de reinicio
normal, no de inyección de fallos.
