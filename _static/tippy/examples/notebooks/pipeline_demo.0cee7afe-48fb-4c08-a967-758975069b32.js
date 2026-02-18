selector_to_html = {"a[href=\"#customize-parameters-optional\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">3. Customize Parameters (Optional)<a class=\"headerlink\" href=\"#customize-parameters-optional\" title=\"Link to this heading\">#</a></h2><p>Modify parameters as needed for your data.</p>", "a[href=\"#run-pipeline\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">4. Run Pipeline<a class=\"headerlink\" href=\"#run-pipeline\" title=\"Link to this heading\">#</a></h2><p>Run the full processing pipeline (motion correction + CNMF).</p>", "a[href=\"#lbm-caiman-pipeline-demo\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">LBM-CaImAn Pipeline Demo<a class=\"headerlink\" href=\"#lbm-caiman-pipeline-demo\" title=\"Link to this heading\">#</a></h1><p>This notebook demonstrates the new unified pipeline API for processing calcium imaging data with CaImAn.</p>", "a[href=\"#evaluation-metrics\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">7. Evaluation Metrics<a class=\"headerlink\" href=\"#evaluation-metrics\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#view-default-parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">2. View Default Parameters<a class=\"headerlink\" href=\"#view-default-parameters\" title=\"Link to this heading\">#</a></h2><p>Check the default CaImAn parameters.</p>", "a[href=\"#setup-paths\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">1. Setup Paths<a class=\"headerlink\" href=\"#setup-paths\" title=\"Link to this heading\">#</a></h2><p>Set your input data path and output directory.</p>", "a[href=\"#load-and-inspect-results\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">5. Load and Inspect Results<a class=\"headerlink\" href=\"#load-and-inspect-results\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#run-caiman-component-evaluation\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">6. Run CaImAn Component Evaluation<a class=\"headerlink\" href=\"#run-caiman-component-evaluation\" title=\"Link to this heading\">#</a></h2><p>Run CaImAn\u2019s <code class=\"docutils literal notranslate\"><span class=\"pre\">evaluate_components</span></code> on existing results to classify components using spatial correlation (r_values) and SNR. This narrows down the 92k raw components to real neurons.</p>"}
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
