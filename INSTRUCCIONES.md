# Guía de Despliegue — Detector de Señales de Tránsito

**Proyecto Final IA — FUP | Tercer Corte**

---

> **IMPORTANTE:** En esta guía verás el texto `TU_USUARIO` en varios lugares. Debes reemplazarlo por el nombre de usuario que elijas al crear tu cuenta en Hugging Face (paso 2.3). Elige un nombre de usuario simple, sin espacios (por ejemplo: `juan123`).

---

## ¿Qué vamos a hacer?

```
Tu PC              Kaggle               HF Hub              HF Spaces           Tu móvil
(código listo)    (entrenamiento)      (guardar modelo)    (app web)           (usar la app)
    │                   │                    │                   │                   │
    │─ notebook ───────►│                    │                   │                   │
    │                   │─ modelo .h5 ──────►│                   │                   │
    │─ carpeta app ──────────────────────────────────────────────►│                   │
    │                   │                    │                   │◄──── navegador ───│
```

- **Kaggle:** plataforma gratuita donde entrenaremos la inteligencia artificial (usando sus GPUs gratis).
- **Hugging Face Hub:** sitio donde guardaremos el modelo entrenado (como un Google Drive para IA).
- **Hugging Face Spaces:** sitio donde desplegaremos la app web (cualquiera puede abrirla desde el móvil).

**Tiempo total estimado: 60–90 minutos** (40 de esos minutos son de espera mientras Kaggle entrena).

---

## Requisitos previos

Antes de empezar, verifica que tienes lo necesario. Abre PowerShell y ejecuta:

```powershell
python --version
```
Deberías ver algo como `Python 3.10.x` o superior.

```powershell
git --version
```
Deberías ver algo como `git version 2.x.x`.

Si alguno de los dos falla, instálalo antes de continuar.

---

## FASE A — Crear las cuentas *(~15 min)*

### Paso 1 — Crear cuenta en Kaggle

1. Ve a: **https://www.kaggle.com/account/login**
2. Haz clic en **"Register with Google"** (más rápido) o en "Email" si prefieres.
3. Completa el registro.
4. **Verifica tu teléfono** (Kaggle lo exige para usar GPU gratis):
   - Una vez dentro, ve a tu foto de perfil (arriba a la derecha) → **"Settings"**
   - Busca la sección **"Phone Verification"**
   - Haz clic en **"Verify"**, pon tu número y sigue las instrucciones
   - ✅ Cuando termine verás: *"Phone number verified"*

### Paso 2 — Crear cuenta en Hugging Face

1. Ve a: **https://huggingface.co/join**
2. Elige un nombre de usuario simple (sin espacios ni tildes, ej: `juan123`) — este será tu `TU_USUARIO`.
3. Completa el registro con tu correo.
4. Verifica tu correo si te lo pide.

### Paso 3 — Crear un token de acceso en Hugging Face

El token es como una contraseña especial para que la terminal pueda subir archivos a tu cuenta.

1. Estando en Hugging Face, haz clic en tu foto (arriba a la derecha) → **"Settings"**
2. En el menú izquierdo haz clic en **"Access Tokens"**
3. Haz clic en el botón **"New token"**
4. Configura así:
   - **Name:** `traffic-signs-deploy`
   - **Type:** selecciona **"Write"** (importante: debe ser Write, no Read)
5. Haz clic en **"Generate a token"**
6. Verás un código largo que empieza con `hf_...` — **cópialo y guárdalo en un bloc de notas**, porque solo se muestra una vez.

---

## FASE B — Entrenar el modelo en Kaggle *(~40 min de espera)*

### Paso 4 — Crear el notebook en Kaggle

1. Ve a **https://www.kaggle.com** e inicia sesión.
2. Haz clic en el botón azul **"Create"** (arriba a la derecha) → selecciona **"New Notebook"**.
3. Se abrirá un editor de código. En el **panel derecho**, haz clic en **"Session options"** (puede aparecer como un ícono de engranaje ⚙️) y cambia:
   - **Accelerator:** selecciona **"GPU T4 x2"**
   - Guarda el cambio.

### Paso 5 — Agregar el dataset GTSRB

1. En el panel derecho, busca la sección **"Input"** y haz clic en **"Add data"**.
2. En el buscador escribe: `gtsrb-german-traffic-sign`
3. Aparecerá el dataset de "Meow Meow" — haz clic en **"Add"**.
4. ✅ Verás que aparece en la lista de datos del panel derecho.

### Paso 6 — Subir el notebook de entrenamiento

1. En el menú del notebook, haz clic en **"File"** → **"Import Notebook"**
2. Selecciona el archivo desde tu PC:
   ```
   D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\notebooks\01_train_cnn_kaggle.ipynb
   ```
3. Haz clic en **"Import"**.

### Paso 7 — (Opcional pero recomendado) Agregar el token de HF como secreto

Esto permite que Kaggle suba el modelo directamente a Hugging Face sin que tengas que descargarlo.

1. En el panel derecho, busca **"Add-ons"** → **"Secrets"**
2. Haz clic en **"Add a new secret"**
3. Configura:
   - **Label:** `HF_TOKEN`
   - **Value:** pega tu token `hf_...` que guardaste antes
4. Haz clic en **"Add"**.

### Paso 8 — Ejecutar el entrenamiento

1. Haz clic en **"Run All"** (▶▶ arriba o en el menú "Run" → "Run all").
2. **Espera 30–40 minutos.** Kaggle entrenará la red neuronal con GPU T4 gratis.
3. Mientras esperas, verás los números de accuracy subir en la celda de entrenamiento. Cuando llegue a más del 90%, está yendo bien.
4. Al final, busca la celda de evaluación — deberías ver algo como:
   ```
   Test Accuracy: 0.9350 (93.5%)
   ```
   Si es ≥ 0.90 (90%), el modelo está bien entrenado.

### Paso 9 — Descargar el modelo entrenado

**Si agregaste el secreto HF_TOKEN (paso 7):** el modelo ya se subió automáticamente a Hugging Face Hub. Ve directo a la Fase C paso 11 para verificarlo.

**Si NO agregaste el secreto:**
1. En el panel derecho, busca la sección **"Output"**.
2. Verás el archivo `cnn_gtsrb_v1.h5` — haz clic en los tres puntos (⋯) → **"Download"**.
3. Guarda el archivo en:
   ```
   D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\models\cnn_gtsrb_v1.h5
   ```

---

## FASE C — Subir el modelo a Hugging Face Hub *(~5 min)*

*(Si el modelo ya se subió automáticamente en el paso 7, pasa directamente al **Paso 11**.)*

### Paso 10 — Subir el modelo desde la terminal

Abre PowerShell **en la carpeta del proyecto**. Para hacerlo rápido: en el Explorador de Windows, navega a `D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO`, y en la barra de direcciones escribe `powershell` y presiona Enter.

Ejecuta estos comandos **uno por uno**:

```powershell
# Instalar la herramienta de Hugging Face
pip install huggingface-hub
```

```powershell
# Iniciar sesión (te pedirá el token; pégalo y presiona Enter)
huggingface-cli login
```
> Cuando te diga `Enter your token:`, pega el token `hf_...` que guardaste y presiona Enter. No verás los caracteres mientras escribes — es normal.

```powershell
# Crear el repositorio del modelo en HF (reemplaza TU_USUARIO)
huggingface-cli repo create traffic-signs-cnn --type model
```
> Cuando pregunte si deseas continuar, escribe `y` y presiona Enter.

```powershell
# Subir el archivo del modelo (reemplaza TU_USUARIO)
huggingface-cli upload TU_USUARIO/traffic-signs-cnn models/cnn_gtsrb_v1.h5
```
> Verás una barra de progreso. Cuando termine, dirá algo como `"cnn_gtsrb_v1.h5" committed`.

### Paso 11 — Verificar que el modelo está en Hugging Face

Abre en el navegador: `https://huggingface.co/TU_USUARIO/traffic-signs-cnn`

✅ Deberías ver la página del modelo con el archivo `cnn_gtsrb_v1.h5` listado.

> **Si el repo es privado:** haz clic en "Settings" del repositorio → busca "Change visibility" → cámbialo a **Public**. La app necesita poder descargarlo.

---

## FASE D — Desplegar la app en Hugging Face Spaces *(~10 min)*

### Paso 12 — Crear el Space en Hugging Face

1. En Hugging Face, haz clic en el botón **"New"** (arriba a la derecha) → **"Space"**.
2. Configura así:
   - **Owner:** tu usuario
   - **Space name:** `traffic-signs-app`
   - **License:** MIT
   - **Space SDK:** selecciona **"Gradio"** ← importante
   - **Space hardware:** **"CPU basic · Free"**
3. Deja las demás opciones en su valor por defecto.
4. Haz clic en **"Create Space"**.
5. Se creará un repositorio vacío — eso es normal por ahora.

### Paso 13 — Actualizar el código con tu nombre de usuario

Antes de subir la app, debes reemplazar `jarolmedina41` por **TU_USUARIO** en dos archivos:

**Archivo 1:** `D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\hf_space\utils.py`

Abre el archivo y cambia la línea:
```python
HF_MODEL_REPO = "jarolmedina41/traffic-signs-cnn"
```
por:
```python
HF_MODEL_REPO = "TU_USUARIO/traffic-signs-cnn"
```

**Archivo 2:** `D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\hf_space\app.py`

Busca la línea:
```python
MODEL_PATH = hf_hub_download(repo_id="jarolmedina41/traffic-signs-cnn", filename="cnn_gtsrb_v1.h5")
```
Cámbiala a:
```python
MODEL_PATH = hf_hub_download(repo_id="TU_USUARIO/traffic-signs-cnn", filename="cnn_gtsrb_v1.h5")
```

### Paso 14 — Subir la app al Space

Abre PowerShell y ejecuta los siguientes comandos:

```powershell
# Ir al escritorio (o cualquier carpeta fuera del proyecto)
cd $env:USERPROFILE\Desktop

# Clonar el Space vacío que acabas de crear (reemplaza TU_USUARIO)
git clone https://huggingface.co/spaces/TU_USUARIO/traffic-signs-app

# Entrar a la carpeta del Space
cd traffic-signs-app
```

```powershell
# Copiar todo el contenido de hf_space/ al Space
Copy-Item "D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\hf_space\*" . -Recurse -Force
```

```powershell
# Preparar y subir los archivos
git add .
git commit -m "Deploy: Detector de Senales de Transito con CNN y BERT"
git push
```
> Si Git te pide usuario y contraseña, usa tu usuario de HF y el token `hf_...` como contraseña.

### Paso 15 — Esperar el build del Space

1. Ve a: `https://huggingface.co/spaces/TU_USUARIO/traffic-signs-app`
2. Haz clic en la pestaña **"Logs"** del Space.
3. Verás el proceso de instalación de paquetes. Espera 3–8 minutos.
4. Cuando veas los mensajes de "Todos los modelos cargados correctamente" y Gradio iniciando, la app está lista.
5. ✅ La app aparecerá en la pestaña **"App"** del Space.

---

## FASE E — Probar desde el móvil *(~2 min)*

### Paso 16 — Abrir la app en el teléfono

1. Abre el navegador de tu teléfono (Chrome o Safari).
2. Ve a: `https://huggingface.co/spaces/TU_USUARIO/traffic-signs-app`
3. La app cargará — tiene dos pestañas:
   - **"Detectar señal"**
   - **"Preguntar al agente"**

### Paso 17 — Probar la detección de señales

1. En la pestaña **"Detectar señal"**, verás un botón de cámara.
2. Haz clic en él y permite el acceso a la cámara cuando el navegador lo pida.
3. Toma una foto de una señal de tránsito (o sube una imagen de señal desde tu galería).
4. ✅ En pocos segundos verás el nombre de la señal y el porcentaje de confianza.

### Paso 18 — Probar el agente Q&A

1. Toca la pestaña **"Preguntar al agente"**.
2. Escribe una pregunta como:
   - `¿Qué significa esta señal?`
   - `¿Qué debo hacer cuando la veo?`
   - `¿Dónde suelen poner esta señal?`
3. Presiona Enter o el botón "Enviar".
4. ✅ El agente responderá basándose en la señal que detectaste antes.

---

## Solución de problemas comunes

| Problema | Solución |
|---|---|
| "GPU no disponible en Kaggle" | Verifica tu teléfono primero (Paso 1), luego intenta de nuevo |
| "Acceso denegado" al subir a HF | Tu token no tiene permisos Write. Ve a HF → Settings → Access Tokens → genera uno nuevo con tipo Write |
| El build del Space falla con error de versiones | Revisa la pestaña "Logs" del Space — si hay error de TensorFlow, escríbeme con el mensaje exacto |
| "File not found: cnn_gtsrb_v1.h5" en la app | El repo del modelo está en privado. Ve a HF → tu modelo → Settings → cambia a Public |
| La cámara no funciona en el móvil | HF Spaces ya usa HTTPS, así que debería funcionar. Asegúrate de dar permisos de cámara al navegador |
| Kaggle dice que los secretos no están disponibles | El notebook no los encontró — descarga el modelo manualmente del panel Output (Paso 9) |
| El Space dice "Building..." por más de 15 min | Algo falló silenciosamente. Ve a Logs y busca el error en rojo |

---

## Resumen de URLs finales

Una vez completado todo, anota estas URLs para tu entrega:

- **App web (acceso móvil):** `https://huggingface.co/spaces/TU_USUARIO/traffic-signs-app`
- **Modelo entrenado:** `https://huggingface.co/TU_USUARIO/traffic-signs-cnn`
- **Código del proyecto:** carpeta local `D:\Documentos\FUP\IA\TALLER\TERCER CORTE\PROYECTO\`
- **Notebook de entrenamiento:** `notebooks/01_train_cnn_kaggle.ipynb` (para mostrar al evaluador)
