// Give a page of two panels a gap wide enough for the right panel's row labels.
//
// Plotly places a subplot by a domain, which is a fraction of the figure's
// width, and draws a row label at whatever width the text comes out to, which
// is a number of pixels. The two agree at one figure width only: below it the
// right panel's labels run back over the left panel's plot area, and above it
// the gap is wider than anything is in. So the gap is measured here, after
// every draw, and set to what the labels in it actually need.
(function () {
  // daylight between a row label and the panel it sits beside
  var SLACK = 8;

  // half a pixel of movement is not worth a redraw, and redrawing is what
  // brings us back here: without a threshold this is a loop
  var SETTLED = 0.5;

  function suffix(column) {
    return column === 1 ? "" : String(column);
  }

  // how many pixels a column's row labels take up to the left of the panel
  // they label: the width of the widest one plus plotly's own tick standoff.
  // Read off the drawn text rather than estimated, since the labels carry
  // markup and are drawn in the browser's own font
  function reachOf(gd, column, layout) {
    var ticks = gd.querySelectorAll("g.y" + suffix(column) + "tick text");
    if (!ticks.length) return 0;

    var size = layout._size;
    var edge =
      gd.getBoundingClientRect().left +
      size.l +
      layout["xaxis" + suffix(column)].domain[0] * size.w;
    var reach = 0;
    Array.prototype.forEach.call(ticks, function (node) {
      var rect = node.getBoundingClientRect();
      if (rect.width) reach = Math.max(reach, edge - rect.left);
    });
    return reach;
  }

  function fit(gd) {
    var layout = gd._fullLayout;
    // one panel keeps its labels in the figure's own margin, which plotly
    // already widens to hold them
    if (!layout || !layout._size || !layout._size.w || !layout.xaxis2) return true;

    var columns = 1;
    while (layout["xaxis" + (columns + 1)]) columns += 1;

    // the widest of the inner columns, so that one gap serves them all and the
    // panels stay the same width as each other
    var reach = 0;
    for (var column = 2; column <= columns; column += 1) {
      reach = Math.max(reach, reachOf(gd, column, layout));
    }
    if (!reach) return false;

    var size = layout._size;
    var gap = (reach + SLACK) / size.w;
    var width = (1 - (columns - 1) * gap) / columns;
    // narrower than the labels themselves, where there is no arrangement left
    // to make: leave the figure as plotly drew it
    if (width <= 0) return true;

    var update = {};
    var moved = false;
    for (column = 1; column <= columns; column += 1) {
      var start = (column - 1) * (width + gap);
      var domain = [start, start + width];
      var current = layout["xaxis" + suffix(column)].domain;
      if (
        Math.abs(current[0] - domain[0]) * size.w > SETTLED ||
        Math.abs(current[1] - domain[1]) * size.w > SETTLED
      ) {
        moved = true;
      }
      update["xaxis" + suffix(column) + ".domain"] = domain;
    }
    if (moved) Plotly.relayout(gd, update);
    return true;
  }

  // the labels are redrawn at every width, so the gap is retaken after each
  // plot rather than once
  function start() {
    var gd = document.querySelector(".plotly-graph-div");
    if (!gd || !window.Plotly || !gd.on) {
      window.setTimeout(start, 60);
      return;
    }
    // says so in the console rather than failing silently, since the measuring
    // reads plotly's own markup and computed layout
    if (!fit(gd)) {
      console.warn("panel gap: no row labels found beside the inner panels; the gap stands");
    }
    gd.on("plotly_afterplot", function () {
      fit(gd);
    });
  }

  start();
})();
