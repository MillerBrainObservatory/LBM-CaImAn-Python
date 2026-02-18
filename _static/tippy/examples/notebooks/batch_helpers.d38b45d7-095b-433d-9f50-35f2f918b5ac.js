selector_to_html = {"a[href=\"#batch-helpers\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Batch Helpers<a class=\"headerlink\" href=\"#batch-helpers\" title=\"Link to this heading\">#</a></h1><p>A notebook to help manage batch results</p>", "a[href=\"#selectively-remove-rows-by-index\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.3. </span>Selectively remove rows by index<a class=\"headerlink\" href=\"#selectively-remove-rows-by-index\" title=\"Link to this heading\">#</a></h2><p>Create a list of indices to remove. You can do this manually</p><p>Some useful ways to create a list:</p>", "a[href=\"#remove-rows-with-no-output-accidental-entries\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.2. </span>Remove rows with no output (accidental entries)<a class=\"headerlink\" href=\"#remove-rows-with-no-output-accidental-entries\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#change-correct-input-movie-path\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.6. </span>Change/correct <code class=\"docutils literal notranslate\"><span class=\"pre\">input_movie_path</span></code><a class=\"headerlink\" href=\"#change-correct-input-movie-path\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#convert-intermediate-mmmap-to-zarr\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.7. </span>Convert intermediate <code class=\"docutils literal notranslate\"><span class=\"pre\">.mmmap</span></code> to <code class=\"docutils literal notranslate\"><span class=\"pre\">.zarr</span></code><a class=\"headerlink\" href=\"#convert-intermediate-mmmap-to-zarr\" title=\"Link to this heading\">#</a></h2><p>TODO: Move these to a separate \u2018helpers\u2019 notebook as they aren\u2019t related specifically to batch management</p>", "a[href=\"#manage-batch-and-dataframe-filepath-locations\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.1. </span>Manage batch and dataframe filepath locations<a class=\"headerlink\" href=\"#manage-batch-and-dataframe-filepath-locations\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#add-a-comment\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.4. </span>Add a <code class=\"docutils literal notranslate\"><span class=\"pre\">comment</span></code><a class=\"headerlink\" href=\"#add-a-comment\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#change-the-item-name\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.5. </span>Change the <code class=\"docutils literal notranslate\"><span class=\"pre\">item_name</span></code><a class=\"headerlink\" href=\"#change-the-item-name\" title=\"Link to this heading\">#</a></h2>"}
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
