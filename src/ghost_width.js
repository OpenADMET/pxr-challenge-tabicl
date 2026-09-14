// Place a figure on a Ghost post, at a plot width every figure on the post
// shares, centred at full window width and kept at that size while there is room
//
// Ghost lays a post body out as a CSS grid of nested columns, main (the 720px
// text column), wide and full, and a card sits in main unless it carries a
// width class. Main is too narrow for these figures, whose row labels can take
// more than half of it.
//
// Widening the card is not enough on its own, because the labels make a figure
// lopsided: they fill a wide left margin against a 30px right one, so a figure
// whose box is centred has its plot and its axis title pushed right of the text
// by half the difference. So the card claims the full column, and the figure is
// put in a frame sized and offset against the plot area rather than the box.
//
// Every figure on the post gets the same plot width, so a reader comparing one
// figure's axis with the next is comparing like with like. That width is the
// widest plot the figure with the widest margins can centre on the text column
// when the window fills the screen, capped at the theme's wide column; a window
// wider than the screen, as on a second monitor, is its own reference. At full
// width each plot is centred. In a narrower window a figure keeps that width
// and moves off centre as a gutter pushes on it, and only once it meets both
// gutters does its plot give up width
(function () {
  // the script tag running this copy, read now because it is null inside any
  // callback; a post can carry several figures, each followed by its own copy
  var SELF = document.currentScript;

  // Ghost's grid container, which the theme hangs the column tracks off
  var CANVAS = ".gh-canvas";

  // raised when a figure's margins change, which can change the shared width
  var MARGINS_EVENT = "ghost-figure-margins";

  // a change smaller than this is not worth a redraw, and every redraw is
  // seen again by the watch on the figure, so without a threshold this is a loop
  var SETTLED = 0.5;

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

  // the ancestor of the figure that the grid lays out: the pasted card's own
  // outer element, since Ghost writes an HTML card's markup without a wrapper
  function gridItem(gd, canvas) {
    var item = gd;
    while (item.parentNode && item.parentNode !== canvas) {
      item = item.parentNode;
    }
    return item.parentNode === canvas ? item : null;
  }

  // how wide an element comes out in the grid, measured with a throwaway one,
  // since the theme gives the gutter as a clamp() and the wide column as a track
  function measure(canvas, className, css) {
    var probe = document.createElement("div");
    probe.className = className;
    probe.style.cssText = css + "height:0;margin:0;padding:0;visibility:hidden;";
    canvas.appendChild(probe);
    var width = probe.getBoundingClientRect().width;
    canvas.removeChild(probe);
    return width;
  }

  // the graphs laid out in this post's grid, every one of which the shared width
  // has to hold
  function postFigures(canvas) {
    return Array.prototype.filter.call(document.querySelectorAll(".plotly-graph-div"), function (graph) {
      return canvas.contains(graph);
    });
  }

  function allDrawn(canvas) {
    return postFigures(canvas).every(function (graph) {
      return graph._fullLayout && graph._fullLayout._size;
    });
  }

  // the widest label or right margin of any figure on the post, which decides
  // how wide a plot every figure can centre
  function widestMargin(canvas) {
    return postFigures(canvas).reduce(function (widest, graph) {
      var size = graph._fullLayout && graph._fullLayout._size;
      return size ? Math.max(widest, size.l, size.r) : widest;
    }, 0);
  }

  // the frame's width and left offset within the card, for the figure as drawn
  function placement(gd, canvas, item) {
    var size = gd._fullLayout._size;
    var page = canvas.getBoundingClientRect();
    var card = item.getBoundingClientRect();
    var available = card.width;
    var centre = page.left + page.width / 2 - card.left;
    var cap = measure(canvas, "kg-width-wide", "");
    var gutter = measure(canvas, "", "width:var(--container-gap);");

    // a theme without the gutter variable measures the whole column instead
    if (!(gutter > 0 && gutter < available / 4)) gutter = 0;

    // the widest plot every figure on the post could centre with the window at
    // full width, where both sides of the centre hold half the plot and the
    // widest margin; anything the page does not span, such as a scrollbar, is
    // taken off the screen too
    var unspanned = window.innerWidth - page.width;
    var reference = Math.max(page.width, (window.screen.availWidth || 0) - unspanned);
    var target = Math.min(cap, reference - 2 * (gutter + widestMargin(canvas)));

    // that plot, or less only where this figure's room between the gutters runs out
    var plot = Math.max(0, Math.min(target, available - 2 * gutter - size.l - size.r));
    var width = Math.min(size.l + plot + size.r, available - 2 * gutter);

    // centred on the text column where that width fits, and otherwise held
    // against whichever gutter it would cross
    var offset = centre - size.l - plot / 2;
    offset = Math.max(gutter, Math.min(offset, available - gutter - width));
    return { width: width, offset: offset };
  }

  function start() {
    var gd = ownFigure();
    if (!gd || !window.Plotly || !gd.on || !gd._fullLayout || !gd._fullLayout._size) {
      window.setTimeout(start, 60);
      return;
    }

    var canvas = gd.closest && gd.closest(CANVAS);
    // every page that is not a Ghost post, including the standalone file this
    // is written as, where the figure already has the window to itself
    if (!canvas) return;

    // the shared width is read off every figure's margins, and a figure later in
    // the post is not even parsed when this copy first runs
    if (document.readyState === "loading" || !allDrawn(canvas)) {
      window.setTimeout(start, 60);
      return;
    }

    var item = gridItem(gd, canvas);
    if (!item) return;
    item.classList.add("kg-width-full");

    // a frame between the card and the figure, which the figure fills
    var frame = document.createElement("div");
    frame.style.cssText = "position:relative;";
    gd.parentNode.insertBefore(frame, gd);
    frame.appendChild(gd);

    var current = { width: -1, offset: -1 };
    var margin = null;
    function update() {
      // a change in this figure's margins can move the width every figure shares,
      // so the others are told; told again with nothing changed, nobody moves
      var own = Math.max(gd._fullLayout._size.l, gd._fullLayout._size.r);
      if (own !== margin) {
        margin = own;
        window.dispatchEvent(new CustomEvent(MARGINS_EVENT));
      }

      var next = placement(gd, canvas, item);
      var moved =
        Math.abs(next.width - current.width) > SETTLED ||
        Math.abs(next.offset - current.offset) > SETTLED;
      if (!moved) return;
      current = next;
      frame.style.width = next.width + "px";
      frame.style.marginLeft = next.offset + "px";
      // drawn afresh at the new width rather than resized. A resize keeps a
      // legend laid out for the old width wherever the new one changes how
      // many rows it wraps to, and repairing that afterwards races the
      // resize's own delayed redraw on a post carrying several figures
      Plotly.newPlot(gd, gd.data, gd.layout, gd._context);
    }

    update();
    afterRedraw(gd, update);
    // a window resize moves the page centre and gutters, and if the frame does
    // not change width plotly redraws nothing for the watch to see
    window.addEventListener("resize", update);
    window.addEventListener(MARGINS_EVENT, update);
  }

  start();
})();
