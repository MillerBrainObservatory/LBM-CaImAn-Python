selector_to_html = {"a[href=\"#preview-raw-traces\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">25.2. </span>Preview \u2018raw traces\u2019<a class=\"headerlink\" href=\"#preview-raw-traces\" title=\"Permalink to this heading\">#</a></h2><p>Show the raw trace for a given pixel by clicking on that pixel.</p>", "a[href=\"#smoothing-with-gaussian-filter\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">25.1. </span>Smoothing with Gaussian Filter<a class=\"headerlink\" href=\"#smoothing-with-gaussian-filter\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#fastplotlib-examples\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">25. </span>Fastplotlib Examples<a class=\"headerlink\" href=\"#fastplotlib-examples\" title=\"Permalink to this heading\">#</a></h1><h2><span class=\"section-number\">25.1. </span>Smoothing with Gaussian Filter<a class=\"headerlink\" href=\"#smoothing-with-gaussian-filter\" title=\"Permalink to this heading\">#</a></h2>"}
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
