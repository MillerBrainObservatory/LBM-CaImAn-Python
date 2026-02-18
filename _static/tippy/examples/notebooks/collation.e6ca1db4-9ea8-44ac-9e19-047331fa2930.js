selector_to_html = {"a[href=\"#z-stack-alignment\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.4. </span>z-stack alignment<a class=\"headerlink\" href=\"#z-stack-alignment\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#total-number-of-neurons\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.2. </span>Total number of Neurons<a class=\"headerlink\" href=\"#total-number-of-neurons\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#align-each-plane-to-the-reference-plane\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.5. </span>1. Align each plane to the reference plane<a class=\"headerlink\" href=\"#align-each-plane-to-the-reference-plane\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#get-outputs-for-each-z-plane\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.1. </span>Get outputs for each z-plane<a class=\"headerlink\" href=\"#get-outputs-for-each-z-plane\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#combine-results-of-different-z-planes\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.3. </span>Combine results of different z_planes<a class=\"headerlink\" href=\"#combine-results-of-different-z-planes\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#id1\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5.6. </span>2. Align each plane to the reference plane<a class=\"headerlink\" href=\"#id1\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#collate-planes\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">5. </span>Collate Planes<a class=\"headerlink\" href=\"#collate-planes\" title=\"Link to this heading\">#</a></h1><p>In this notebook, we merge CNMF results across each of our z-planes.</p>"}
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
