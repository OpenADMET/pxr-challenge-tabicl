// Make a row's axis label hover the row.
//
// Plotly gives an axis tick no hover event of its own, so hovering a label is
// turned into a hover on the point that label names. The tooltip that opens is
// the point's own, so a label and its marker say exactly the same thing.
(function () {
  // plotly wraps the content of a tag like <sub> in zero-width characters when
  // it renders the label, so the tick's text and the trace's own value only
  // agree once both are stripped of markup and of those
  function plain(text) {
    return (text || "")
      .replace(/<[^>]*>/g, "")
      .replace(/[\u200b\u200c\u200d\ufeff]/g, "")
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

  function wire(gd) {
    var count = 0;
    var byLabel = pointsByLabel(gd);
    var ticks = gd.querySelectorAll(
      [
        "g.yaxislayer-above text",
        "g.yaxislayer-below text",
        "g.ytick text",
        "text.ytick",
        "g[class*='yaxislayer'] text"
      ].join(", ")
    );
    Array.prototype.forEach.call(ticks, function (node) {
      if (node.dataset.rowHover) return;
      node.dataset.rowHover = "1";
      // the axis layers do not take pointer events, which is why a tick has no
      // hover of its own to begin with; the label and the group holding it
      // both have to be opted back in before a listener on them can fire
      node.style.pointerEvents = "all";
      if (node.parentNode && node.parentNode.style) {
        node.parentNode.style.pointerEvents = "all";
      }
      node.style.cursor = "pointer";
      node.addEventListener("mouseenter", function () {
        var label = plain(node.textContent);
        var subplot = subplotOfNode(node, gd);
        var hit = byLabel[subplot + " " + label];
        if (!hit) {
          // a tick whose subplot could not be read still finds its row when
          // the label is unique on the page
          Object.keys(byLabel).forEach(function (key) {
            if (!hit && key.slice(key.indexOf(" ") + 1) === label) hit = byLabel[key];
          });
        }
        if (!hit) return;
        Plotly.Fx.hover(
          gd,
          [{ curveNumber: hit.curveNumber, pointNumber: hit.pointNumber }],
          hit.subplot
        );
      });
      node.addEventListener("mouseleave", function () {
        Plotly.Fx.unhover(gd);
      });
      count += 1;
    });
    return count;
  }

  // the ticks are redrawn on resize, so the wiring is redone after each plot
  function start() {
    var gd = document.querySelector(".plotly-graph-div");
    if (!gd || !window.Plotly || !gd.on) {
      window.setTimeout(start, 60);
      return;
    }
    var wired = wire(gd);
    // says so in the console rather than failing silently, since the selectors
    // depend on plotly's own markup
    if (!wired) {
      console.warn("row hover: no axis tick labels matched; labels stay inert");
    }
    gd.on("plotly_afterplot", function () {
      wire(gd);
    });
  }

  start();
})();
