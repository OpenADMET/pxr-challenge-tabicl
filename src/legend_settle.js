// Put a figure's legend back above its plot after the figure changes width.
//
// Every figure here draws a horizontal legend anchored by its bottom edge to
// the top of the plot. When the figure is narrow enough for that legend to
// wrap, plotly lays it out in two rows and grows the top margin to hold them;
// when the figure then widens, the margin shrinks back to one row but the
// legend keeps the offset it had for two, and sits a row deep in the plot.
// Narrowing leaves the opposite mistake, a legend floating clear of it.
// Plotly's resize does not recompute that offset, and neither does relayout of
// the size. Setting the legend's position again, even to the value it already
// has, does.
//
// It happens whenever plotly's own resize widens a figure, which is what a
// reader enlarging the window does; placement on a Ghost post draws the
// figure afresh instead, and never leaves the legend stale to begin with
(function () {
  // the script tag running this copy, read now because it is null inside any
  // callback; a post can carry several figures, each followed by its own copy
  var SELF = document.currentScript;

  // how far a legend's bottom edge may sit from the plot's top edge, either
  // way, before it is stale
  var TOLERANCE = 1.5;

  // the figure this copy follows: the last graph before its script tag
  function ownFigure() {
    var graphs = document.querySelectorAll(".plotly-graph-div");
    var mine = null;
    for (var i = 0; i < graphs.length; i += 1) {
      var before = SELF && (graphs[i].compareDocumentPosition(SELF) & Node.DOCUMENT_POSITION_FOLLOWING);
      if (!SELF || before) mine = graphs[i];
    }
    return mine;
  }

  // run a check each time plotly finishes changing the figure. Plotly does not
  // always emit plotly_afterplot after a redraw, so the figure's markup is
  // watched rather than its events; the hover layer is left out, since a
  // tooltip redraws nothing a check reads
  function afterRedraw(gd, check) {
    var pending = false;
    new MutationObserver(function (records) {
      var drawn = records.some(function (record) {
        var node = record.target;
        return !(node.closest && node.closest(".hoverlayer"));
      });
      if (!drawn || pending) return;
      pending = true;
      window.requestAnimationFrame(function () {
        pending = false;
        check();
      });
    }).observe(gd, { childList: true, subtree: true });
  }

  // whether a legend meant to rest on the plot's top edge has come away from it
  function isStale(gd) {
    var layout = gd._fullLayout;
    var legend = layout && layout.legend;
    var node = gd.querySelector("g.legend");
    if (!legend || !node || legend.yanchor !== "bottom" || legend.y < 1) return false;
    var plotTop = gd.getBoundingClientRect().top + layout._size.t;
    return Math.abs(node.getBoundingClientRect().bottom - plotTop) > TOLERANCE;
  }

  function start() {
    var gd = ownFigure();
    if (!gd || !window.Plotly || !gd.on || !gd._fullLayout || !gd._fullLayout._size) {
      window.setTimeout(start, 60);
      return;
    }

    // one attempt per width, so a legend plotly cannot place is left alone
    // rather than relaid out forever
    var triedAt = null;
    function settle() {
      var width = gd._fullLayout.width;
      if (width === triedAt || !isStale(gd)) return;
      triedAt = width;
      Plotly.relayout(gd, { "legend.y": gd._fullLayout.legend.y });
    }

    settle();
    afterRedraw(gd, settle);
  }

  start();
})();
