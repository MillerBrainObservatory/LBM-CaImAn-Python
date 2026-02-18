selector_to_html = {"a[href=\"#first-steps\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">First Steps<a class=\"headerlink\" href=\"#first-steps\" title=\"Link to this heading\">#</a></h1><h2>Input/Output Directories<a class=\"headerlink\" href=\"#input-output-directories\" title=\"Link to this heading\">#</a></h2><p>Each time you run an algorithm, the results are saved to disk. This is your <code class=\"docutils literal notranslate\"><span class=\"pre\">batch_path</span></code>.\nIt tracks the input file-path at the time you run the algorithm.\nHowever, its often helpful to move results.</p><p>To allow this, we call <code class=\"docutils literal notranslate\"><span class=\"pre\">mc.set_parent_raw_data_path(/path/to/raw.tiff)</span></code>.\nNow, our results are saved <strong>relative to this location</strong>.</p>", "a[href=\"#input-output-directories\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Input/Output Directories<a class=\"headerlink\" href=\"#input-output-directories\" title=\"Link to this heading\">#</a></h2><p>Each time you run an algorithm, the results are saved to disk. This is your <code class=\"docutils literal notranslate\"><span class=\"pre\">batch_path</span></code>.\nIt tracks the input file-path at the time you run the algorithm.\nHowever, its often helpful to move results.</p><p>To allow this, we call <code class=\"docutils literal notranslate\"><span class=\"pre\">mc.set_parent_raw_data_path(/path/to/raw.tiff)</span></code>.\nNow, our results are saved <strong>relative to this location</strong>.</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
