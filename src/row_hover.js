// Make a row's axis label hover the row.
//
// Plotly gives an axis tick no hover event of its own, so hovering a label is
// turned into a hover on the point that label names. The tooltip that opens is
// the point's own, so a label and its marker say exactly the same thing.
//
// Nothing is attached to the labels themselves. Plotly replaces every label
// node when it redraws, and a redraw is not always followed by an event this
// script would see, so listeners on the nodes are lost without warning. The
// figure listens instead, and a stylesheet rule opts the labels back into
// pointer events, which holds for whatever nodes plotly draws next
(function () {
  // the script tag running this copy, read now because it is null inside any
  // callback; a post can carry several figures, each followed by its own copy
  var SELF = document.currentScript;

  // the y axis labels, under each of the class names plotly has used for them
  var TICKS = [
    "g.yaxislayer-above text",
    "g.yaxislayer-below text",
    "g.ytick text",
    "text.ytick",
    "g[class*='yaxislayer'] text"
  ].join(", ");

  // one rule for the whole page, however many figures carry a copy of this
  var STYLE_ID = "row-hover-style";

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

  // plotly wraps the content of a tag like <sub> in zero-width characters when
  // it renders the label, so the tick's text and the trace's own value only
  // agree once both are stripped of markup and of those
  function plain(text) {
    return (text || "")
      .replace(/<[^>]*>/g, "")
      .replace(/[​‌‍﻿]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function subplotOfTrace(trace) {
    return (trace.xaxis || "x") + (trace.yaxis || "y");
  }

  // a tick belongs to one subplot, which figure 2 needs: two panels on a page
  // can hold the same label and only one of them is under the cursor
  function subplotOfNode(node, root) {
    for (var el = node; el && el !== root; el = el.parentNode) {
      var name = el.getAttribute && el.getAttribute("class");
      var found = name && name.match(/(?:^|\s)(x[0-9]*y[0-9]*)(?:\s|$)/);
      if (found) return found[1];
    }
    return null;
  }

  // the markers carry the tooltip; the whisker traces are lines and are skipped
  function pointsByLabel(gd) {
    var byLabel = {};
    (gd.data || []).forEach(function (trace, curve) {
      if (trace.mode !== "markers" || !trace.y) return;
      var subplot = subplotOfTrace(trace);
      Array.prototype.forEach.call(trace.y, function (label, point) {
        var key = subplot + " " + plain(label);
        if (!(key in byLabel)) {
          byLabel[key] = { curveNumber: curve, pointNumber: point, subplot: subplot };
        }
      });
    });
    return byLabel;
  }

  // the axis layers inherit pointer-events: none from plotly's own stylesheet,
  // which is why a label takes no hover to begin with. A label is hit across
  // its whole box rather than only its glyphs, since a label set on two lines
  // leaves a gap between them that the cursor would otherwise fall through; a
  // browser without bounding-box drops that declaration and keeps all
  function allowPointer() {
    if (document.getElementById(STYLE_ID)) return;
    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent =
      ".js-plotly-plot g[class*='yaxislayer'] { pointer-events: all !important; }" +
      " .js-plotly-plot g[class*='yaxislayer'] text { pointer-events: all !important;" +
      " pointer-events: bounding-box !important; cursor: pointer; }";
    document.head.appendChild(style);
  }

  // the label an event happened on, resolved from the tspan plotly puts a
  // <sub> in up to the text element that holds the whole label
  function labelOf(event, gd) {
    var node = event.target;
    var text = node && node.closest ? node.closest(TICKS) : null;
    return text && gd.contains(text) ? text : null;
  }

  function hover(gd, text) {
    var byLabel = pointsByLabel(gd);
    var label = plain(text.textContent);
    var hit = byLabel[subplotOfNode(text, gd) + " " + label];
    // a tick whose subplot could not be read still finds its row when the
    // label is unique on the page
    if (!hit) {
      Object.keys(byLabel).forEach(function (key) {
        if (!hit && key.slice(key.indexOf(" ") + 1) === label) hit = byLabel[key];
      });
    }
    if (!hit) return;
    Plotly.Fx.hover(gd, [{ curveNumber: hit.curveNumber, pointNumber: hit.pointNumber }], hit.subplot);
  }

  function listen(gd) {
    gd.addEventListener("mouseover", function (event) {
      var text = labelOf(event, gd);
      // moving between the pieces of one label is not a new hover
      if (!text || (event.relatedTarget && text.contains(event.relatedTarget))) return;
      hover(gd, text);
    });
    gd.addEventListener("mouseout", function (event) {
      var text = labelOf(event, gd);
      if (!text || (event.relatedTarget && text.contains(event.relatedTarget))) return;
      Plotly.Fx.unhover(gd);
    });
  }

  function start() {
    var gd = ownFigure();
    if (!gd || !window.Plotly || !gd.on) {
      window.setTimeout(start, 60);
      return;
    }

    // a figure two copies both resolved to keeps one pair of listeners
    if (gd.dataset.rowHover) return;
    gd.dataset.rowHover = "1";

    allowPointer();
    listen(gd);

    // says so in the console rather than failing silently, since the selectors
    // depend on plotly's own markup
    if (!gd.querySelector(TICKS)) {
      console.warn("row hover: no axis tick labels matched; labels stay inert");
    }
  }

  start();
})();
