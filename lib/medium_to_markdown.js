const mediumToMarkdown = require('medium-to-markdown');

mediumToMarkdown
  .convertFromUrl(
    'https://towardsdatascience.com/using-kdtree-to-detect-similarities-in-a-multidimensional-dataset-4be276dcf616'
  )
  .then(function (markdown) {
    console.log(markdown); //=> Markdown content of medium post
  });
