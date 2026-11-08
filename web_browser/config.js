// Where the page finds its files when served by serve.py: the export/ bundle and the eval .tsv files next to it.
// web_browser/build_pages.py writes its own config.js for GitHub Pages, pointing at Hugging Face.
window.LYRA_CONFIG = { model: "../export/", examples: "../", cache: null };
