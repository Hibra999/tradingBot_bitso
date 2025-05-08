# (NO SIRVE, NUEVO EN DESAROLLO)

Una aplicación de código abierto para ejecutar estrategias de trading automatizadas utilizando Dash y ccxt. Este bot interactúa con el exchange Bitso y proporciona un dashboard interactivo para visualizar datos y resultados en tiempo real.
---

## Instalación

### Requisitos previos

1. **Python** (3.10.0 o superior): Descárgalo de [python.org](https://www.python.org/).
2. **Editor de código** (opcional pero recomendado): Visual Studio Code, PyCharm u otro de tu preferencia.
3. **Librerías necesarias**: Instálalas con los siguientes comandos en tu terminal:
    ```bash
    pip install dash dash-bootstrap-components plotly pandas numpy ccxt pandas-ta
    ```
    Nota: También se puede instalar `virtualenv` para un entorno aislado.


## Ejecución

1. **Configura y ejecuta el script principal**:
   - Navega al directorio del proyecto y ejecuta el script:
     ```bash
     python b3.py
     ```
2. Abre tu navegador y ve a `http://127.0.0.1:8050/`.

---

## Instrucciones de uso

1. **Selecciona el Token**: En el menú desplegable, selecciona el par de criptomonedas (BTC/MXN, ETH/MXN, etc.) que desees analizar.
2. **Límite de velas**: Ajusta el número de velas históricas que deseas cargar (por defecto: 500).
3. **Estrategia**: Escoge entre las siguientes estrategias:
   - **Original**: Basada en medias móviles y niveles de soporte/resistencia.
   - **NFI**: Utiliza 21 condiciones de compra y 8 condiciones de venta.
   - **Supertrend**: Señales de compra y venta generadas con el indicador Supertrend.
4. Haz clic en "Aplicar Configuración" para cargar los datos y simular las estrategias.
5. Observa el gráfico actualizado en tiempo real, junto con los resultados de victorias, derrotas y empates.

---

## Funcionalidades

1. **Visualización gráfica**:
   - Gráfico interactivo con precios históricos, señales de compra (triángulos verdes) y venta (triángulos rojos).
2. **Indicadores integrados**:
   - RSI, Bollinger Bands, medias móviles, Supertrend, etc.
3. **Resultados en tiempo real**:
   - Balance de dinero y monedas.
   - Número de operaciones exitosas, empates y pérdidas.

---

---

## importante

1. **Simulación vs Trading real**:
   - Por defecto, el bot opera en modo de simulación (`modo_simulacion = True`).
   - Cambia a `modo_simulacion = False` si deseas operar con dinero real. **Advertencia: Asegúrate de entender los riesgos asociados.**
   
2. **Llaves de API**:
   - Si deseas operar en modo real, introduce tus llaves API en la configuración de `ccxt`:
     ```python
     exchange_ccxt = ccxt.bitso({
         'apiKey': 'TU_API_KEY',
         'secret': 'TU_API_SECRET',
         'enableRateLimit': True
     })
     ```

3. **Precisión del código**:
   - Este bot es un punto de partida. Realiza pruebas exhaustivas antes de usarlo en cuentas reales.

---


