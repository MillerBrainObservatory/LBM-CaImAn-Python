selector_to_html = {"a[href=\"#command-line-usage-overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.5. </span>Command Line Usage Overview:<a class=\"headerlink\" href=\"#command-line-usage-overview\" title=\"Link to this heading\">#</a></h2><p>*int = integer, 0 1 2 3 etc.\n*algo = mcorr, cnmf, or cnmfe\n*str=a path of some sort. will be fed into pathlib.Path(str).resolve() to expand <code class=\"docutils literal notranslate\"><span class=\"pre\">~</span></code> chars.</p>", "a[href=\"#examples\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.6. </span>Examples<a class=\"headerlink\" href=\"#examples\" title=\"Link to this heading\">#</a></h2><p>Chain mcorr and cnmf together:</p>", "a[href=\"#batch-utilities\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Batch-Utilities<a class=\"headerlink\" href=\"#batch-utilities\" title=\"Link to this heading\">#</a></h1><p>Before continuing, users should review the <a class=\"reference external\" href=\"https://mesmerize-core.readthedocs.io/en/latest/user_guide.html\">mesmerize-core user guide</a>.</p>", "a[href=\"#add-a-batch-item\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.3. </span>Add a batch item<a class=\"headerlink\" href=\"#add-a-batch-item\" title=\"Link to this heading\">#</a></h2><p>Next, we add an item to the batch.</p><p>A batch item is a combination of:</p>", "a[href=\"#run-a-batch-item\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.4. </span>Run a batch item<a class=\"headerlink\" href=\"#run-a-batch-item\" title=\"Link to this heading\">#</a></h2><p>After adding an item, running the item is as easy as calling <code class=\"docutils literal notranslate\"><span class=\"pre\">row.caiman.run()</span></code>:</p>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.1. </span>Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p>The general workflow is as follows:</p>", "a[href=\"#create-a-batch\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.2. </span>Create a Batch<a class=\"headerlink\" href=\"#create-a-batch\" title=\"Link to this heading\">#</a></h2><p>See <a class=\"reference external\" href=\"https://mesmerize-core.readthedocs.io/en/latest/api/functions.html#mesmerize_core.load_batch\" title=\"(in mesmerize-core)\"><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mesmerize_core.load_batch()</span></code></a>, <a class=\"reference external\" href=\"https://mesmerize-core.readthedocs.io/en/latest/api/functions.html#mesmerize_core.create_batch\" title=\"(in mesmerize-core)\"><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mesmerize_core.create_batch()</span></code></a>,</p>"}
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
