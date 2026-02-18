selector_to_html = {"a[href=\"#run-pipeline\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">3. Run pipeline<a class=\"headerlink\" href=\"#run-pipeline\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#lbm-caiman-pipeline-demo\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">LBM-CaImAn Pipeline Demo<a class=\"headerlink\" href=\"#lbm-caiman-pipeline-demo\" title=\"Link to this heading\">#</a></h1><p>calcium imaging segmentation pipeline using CaImAn CNMF on light-beads microscopy data.</p><p>sections:</p>", "a[href=\"#filter-and-visualize\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">5. Filter and visualize<a class=\"headerlink\" href=\"#filter-and-visualize\" title=\"Link to this heading\">#</a></h2><p>strict AND filtering: a component must pass ALL of:</p>", "a[href=\"#component-evaluation-on-existing-results\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">4. Component evaluation on existing results<a class=\"headerlink\" href=\"#component-evaluation-on-existing-results\" title=\"Link to this heading\">#</a></h2><p>if the pipeline already ran, load saved estimates and run CaImAn\u2019s\n<code class=\"docutils literal notranslate\"><span class=\"pre\">evaluate_components</span></code> to compute r_values (spatial correlation) and\nSNR_comp (signal-to-noise) for each component.</p><p>important: CaImAn\u2019s evaluate_components has a bug in its non-memmap\ncode path (assumes d1,d2,T axis order but movie is T,d1,d2). we\nalways load through a CaImAn-format mmap file to avoid this.</p>", "a[href=\"#load-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">1. Load data<a class=\"headerlink\" href=\"#load-data\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#configure-parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">2. Configure parameters<a class=\"headerlink\" href=\"#configure-parameters\" title=\"Link to this heading\">#</a></h2><p>key parameters:</p>"}
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
