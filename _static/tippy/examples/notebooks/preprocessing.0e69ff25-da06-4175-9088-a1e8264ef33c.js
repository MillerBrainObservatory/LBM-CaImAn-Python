selector_to_html = {"a[href=\"#pre-processing-wip\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Pre-processing (WIP)<a class=\"headerlink\" href=\"#pre-processing-wip\" title=\"Link to this heading\">#</a></h1><h2>Scan-phase correction<a class=\"headerlink\" href=\"#scan-phase-correction\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#percentile-filtering\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Percentile Filtering<a class=\"headerlink\" href=\"#percentile-filtering\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#scan-phase-correction\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Scan-phase correction<a class=\"headerlink\" href=\"#scan-phase-correction\" title=\"Link to this heading\">#</a></h2>"}
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
