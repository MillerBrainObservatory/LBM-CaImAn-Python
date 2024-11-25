selector_to_html = {"a[href=\"#percentile-filtering\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.1. </span>Percentile Filtering<a class=\"headerlink\" href=\"#percentile-filtering\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#threshold-your-movie\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1.3. </span>Threshold your movie<a class=\"headerlink\" href=\"#threshold-your-movie\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#save-as-tiff-zarr\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1.2. </span>Save as <code class=\"docutils literal notranslate\"><span class=\"pre\">.tiff</span></code> / <code class=\"docutils literal notranslate\"><span class=\"pre\">.zarr</span></code><a class=\"headerlink\" href=\"#save-as-tiff-zarr\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#scan-phase-correction\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Scan-phase correction<a class=\"headerlink\" href=\"#scan-phase-correction\" title=\"Permalink to this heading\">#</a></h1><h2><span class=\"section-number\">2.1. </span>Percentile Filtering<a class=\"headerlink\" href=\"#percentile-filtering\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#preprocessing\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Preprocessing<a class=\"headerlink\" href=\"#preprocessing\" title=\"Permalink to this heading\">#</a></h1><h2><span class=\"section-number\">1.1. </span>scanreader<a class=\"headerlink\" href=\"#scanreader\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#correlation-max-images\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1.4. </span>Correlation / Max Images<a class=\"headerlink\" href=\"#correlation-max-images\" title=\"Permalink to this heading\">#</a></h2>", "a[href=\"#scanreader\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1.1. </span>scanreader<a class=\"headerlink\" href=\"#scanreader\" title=\"Permalink to this heading\">#</a></h2>"}
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
