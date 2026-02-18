selector_to_html = {"a[href=\"#segmentation\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">4. </span>Segmentation<a class=\"headerlink\" href=\"#segmentation\" title=\"Link to this heading\">#</a></h1><h2><span class=\"section-number\">4.1. </span>CNMF Parameters<a class=\"headerlink\" href=\"#cnmf-parameters\" title=\"Link to this heading\">#</a></h2><p id=\"decay-times\"><a class=\"reference external\" href=\"https://www.janelia.org/jgcamp8-calcium-indicators\">Decay times</a></p>", "a[href=\"#cnmf-parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">4.1. </span>CNMF Parameters<a class=\"headerlink\" href=\"#cnmf-parameters\" title=\"Link to this heading\">#</a></h2><p id=\"decay-times\"><a class=\"reference external\" href=\"https://www.janelia.org/jgcamp8-calcium-indicators\">Decay times</a></p>"}
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
