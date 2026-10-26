# ROME workshop

[Otevřít notebook v Google Colabu](https://colab.research.google.com/github/Security-FIT/Latium/blob/workshop-version/workshop/clean-notebook.ipynb)
· [clean-notebook.ipynb](clean-notebook.ipynb)

1. Vyberte CUDA GPU runtime a spusťte obě kódové buňky shora dolů.
2. V první buňce ponechte předvyplněný `COVARIANCE_DRIVE_URL`, nebo vložte vlastní
   Google Drive odkaz na kovarianci. Pro soukromý soubor zapněte `AUTHENTICATE_DRIVE`.
3. Otevřete vypsaný odkaz `https://….gradio.live` a klikněte na **Připravit model**.
   Kovarianci můžete místo Drive odkazu nahrát jako `.pt` přímo v Gradio.
4. Vyzkoušejte trasování, připravený edit, vícekrokové otázky, vlastní šablony a GRAM.

**Implementaci ani nastavení checkoutu není potřeba upravovat.** `LOCAL_LATIUM`
a `LATIUM_REF` ponechte na výchozích hodnotách; vlastní fakta zadáváte v Gradio.
Notebook importuje frontend i výpočty z větve `workshop-version`, nekopíruje jejich
implementaci. Výchozí `SEED = 42` lze změnit v první buňce.

Podrobnosti jsou v [návodu k workshopu](../src/workshop/README.md).
