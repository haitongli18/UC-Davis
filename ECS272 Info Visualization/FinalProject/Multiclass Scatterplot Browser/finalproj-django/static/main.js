(function () {
    jQuery.ajax({
        method: "GET",
        url: "get_scatter_points/",
        success: function (d) {
            console.log(d);
            mainScatter(d);
        }
    })
})()

//dimensions for scatter plot
var width = 500;
var height = 500;
var margin = { left: 60, right: 20, top: 20, bottom: 60 };
var defaultRadius = 0.05;
var currentPointSize = defaultRadius;
var mainOpacity = 1;

var svg1 = d3.select('#scatter')
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    ;

var dotg = svg1.append("g")
    .attr("class", "dots");

console.log(svg1);
var mx;
var my;
var axisDiff; // Used for making sure the points are real sized
var level; // zoom level
dots = [];
origDots = [];

function scaleRadius(r) {
    return r * axisDif;
}

function setPointSizeDisplay() {
    d3.select("output#pointsize").text(currentPointSize);
}

//rendering the main scatter plot
function mainScatter(data) {
    //get data
    for (i = 0; i < data.x.length; i++) {
        var dot = {
            x: data.x[i],
            y: data.y[i],
            c: data.c[i],
            c_name: data.c_name[i],
            clr: data.clr[i],
            lvl: data.lvl[i]
        };
        dots.push(dot);
    }

    for (i = 0; i < data.orig_x.length; i++) {
        var dot = {

            x: data.orig_x[i],
            y: data.orig_y[i],
            c: data.orig_c[i],
            c_name: data.orig_c_name[i],
            clr: data.orig_clr[i]
        };
        origDots.push(dot);
    }

    //draw
    // The domain should be shared so that way radius is meaningful in both dims
    shared_domain = [d3.min(dots, d => Math.min(d.x, d.y)), d3.max(dots, d => Math.max(d.x, d.y))];
    mx = d3.scaleLinear()
        .domain(shared_domain)
        .range([0, width + margin.left + margin.right]);
    my = d3.scaleLinear()
        .domain(shared_domain)
        .range([height + margin.top + margin.bottom, 0]);
    axisDif = mx(1) - mx(0);
    level = 1.0;

    setPointSizeDisplay();

    var xAxis = d3.axisBottom(mx);
    var yAxis = d3.axisRight(my);

    var colors = d3.scaleLinear()
        .domain([0, d3.max(data.c)])
        .range(d3.schemeCategory10);

    var scatterPlot = dotg.selectAll('circle')
        .data(dots)
        .enter()
        .append('circle')
        .filter(function (d) { return d.lvl == 1 })
        .attr('cx', d => mx(d.x))
        .attr('cy', d => my(d.y))
        .attr('data-x', d => d.x)
        .attr('data-y', d => d.y)
        .attr("class", "show")
        .attr("id", d => d.c)
        .attr("r", scaleRadius(currentPointSize / level))
        .attr('opacity', mainOpacity)
        .attr("fill", d => d.clr);

    // x axis
    var gX = svg1.append("g")
        .attr("transform", "translate(0," + (height + margin.bottom) + ")")
        .call(xAxis)
        .attr("class", "xAxis")
        .append("text")
        .attr("fill", "#000")
        .attr("x", width / 2)
        .attr('y', margin.bottom / 2)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .text("x");

    // y axis
    var gY = svg1.append("g")
        .call(yAxis)
        .attr("class", "yAxis")
        .append("text")
        .attr("fill", "#000")
        .attr("transform", "rotate(-90)")
        .attr("x", - height / 2)
        .attr("y", - margin.left)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .text("y");

    //small views for different classes
    var c = Array.from(new Set(data.orig_c));
    for (i = 0; i < c.length; i++) {
        var data = origDots.filter(function (el) {
            return el.c == c[i];
        });
        drawScatter(data);
    }

    //point size slider
    d3.select("#pointsize-slider").on("input", function () {
        currentPointSize = this.value;
        setPointSizeDisplay();
        dotg.selectAll("circle.show").attr("r", scaleRadius(currentPointSize / level));
    });

    //zooming
    var zoom = d3.zoom()
        .scaleExtent([1, 4])
        .on("zoom", zoomed);

    $("#zoom").on('click', function () {
        svg1.on(".dragstart", null);
        svg1.on(".drag", null);
        svg1.on(".dragend", null);
        svg1.call(zoom);
    })

    //zoom function
    function zoomed() {
        level = d3.event.transform.k;
        newdata = dots.filter(function (e) {
            return e.lvl <= level;
        })
        d3.select("output#zoomlevel").text(level.toFixed(2));
        d3.select("output#zoomlevel").property("value", level.toFixed(2));

        for (i = 0; i < c.length; i++) {
            console.log(svg1.select(".dots").selectAll("circle[id='" + c[i] + "']"));
            updateScatter(svg1.select(".dots").selectAll("circle[id='" + c[i] + "']"), c[i]);
        }

        svg1.select(".dots").selectAll("circle")
            .attr("r", scaleRadius(currentPointSize / level));

        svg1.select(".dots").attr("transform", d3.event.transform);

        svg1.select(".xAxis").call(xAxis.scale(d3.event.transform.rescaleX(mx)));
        svg1.select(".yAxis").call(yAxis.scale(d3.event.transform.rescaleY(my)));
    }

    //lasso selection
    var lasso_start = function () {
        console.log('start')
        lasso.items()
            .attr("r", 5)
            .classed("not_possible", true)
            .classed("selected", false);
    };

    var lasso_draw = function () {
        console.log('draw')
        lasso.possibleItems()
            .classed("not_possible", false)
            .classed("possible", true);
        lasso.notPossibleItems()
            .classed("not_possible", true)
            .classed("possible", false);
    };

    var lasso_end = function () {
        console.log('end')
        lasso.items()
            .classed("not_possible", false)
            .classed("possible", false);
        lasso.selectedItems()
            .classed("selected", true)
            .attr('fill', d => d.clr)
            .attr("r", 8);
        s = lasso.selectedItems();
        console.log(s._groups[0].length);
        if (s._groups[0].length == 0) {
            lasso.items()
                .attr('fill', d => d.clr);
        } else {
            lasso.notSelectedItems()
                .attr('fill', 'white')
                .attr("r", 5);
        }
    };

    var s = svg1.selectAll('circle');
    const lasso = d3.lasso()
        .closePathDistance(305)
        .closePathSelect(true)
        .targetArea(svg1)
        .items(s)
        .on("start", lasso_start)
        .on("draw", lasso_draw)
        .on("end", lasso_end);

    $("#lasso").on('click', function () {
        svg1.on(".zoom", null);
        svg1.call(lasso);
    })
}

//update function for the main scatter plot
function updateScatter(selection, c) {
    //console.log(selection);
    xs = [];
    selection.each(d => xs.push(d.x));

    ys = [];
    selection.each(d => ys.push(d.y));

    x0 = d3.min(xs);
    x1 = d3.max(xs);
    y0 = d3.min(ys);
    y1 = d3.max(ys);

    var newdata = [];
    var level = d3.select("output#zoomlevel")._groups[0][0].value;
    console.log(level);
    newdata = dots.filter(e => e.x >= x0 && e.x <= x1 && e.y >= y0 && e.y <= y1 && e.lvl <= level && e.c == c);

    var update = svg1.select(".dots").selectAll("circle[id='" + c + "']")
        .data(newdata);

    update
        .join('circle')
        .attr("class", "show")
        .attr("cx", d => mx(d.x))
        .attr("cy", d => my(d.y))
        .attr("fill", d => d.clr)
        .attr("id", d => d.c)
        .attr("r", scaleRadius(0.05 / level))
        .attr('opacity', mainOpacity);

    update.exit().remove();

    console.log(newdata);
}

//rendering small class-specific scatter plots
function drawScatter(origDots) {
    var width = 200;
    var height = 200;
    var margin = { left: 60, right: 60, top: 30, bottom: 60 }

    var svg2 = d3.select('#classes')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .attr("class", origDots[0].c);

    var dotg = svg2.append('g')
        .attr("transform", "translate(" + margin.left + ", " + margin.top + ")")
        .attr("class", "scatter")
        .attr("id", origDots[0].c_name);

    var x = d3.scaleLinear()
        .domain([d3.min(origDots, d => d.x), d3.max(origDots, d => d.x)])
        .range([0, width]);

    var y = d3.scaleLinear()
        .domain([d3.min(origDots, d => d.y), d3.max(origDots, d => d.y)])
        .range([height, 0]);

    console.log(x.domain()[0] + " " + x.domain()[1]);

    svg2.append("foreignObject")
    .attr("width", 50)
    .attr("height", 50)
    .append("xhtml:body")
    .html("<form><input type=checkbox class=checkbox id=" + origDots[0].c_name + " checked/></form>")
    .on("click", function(d, i){
        console.log(svg2.select(".checkbox").node().checked);
        if(!svg2.select(".checkbox").node().checked) {
            svg1.selectAll("circle[id='" + svg2.attr("class") + "']").remove();
        } else {
            updateScatter(svg2.select("g.scatter").selectAll("circle.x_selected.y_selected"), svg2.attr("class"));
        }
    });

    var scatterPlot = dotg.selectAll('circle')
        .data(origDots)
        .enter()
        .append('circle')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('data-x', d => d.x)
        .attr('data-y', d => d.y)
        .attr("data-clr", d => d.clr)
        .attr("r", 5)
        .attr('opacity', 0.5)
        .classed("x_selected", true)
        .classed("y_selected", true)
        .attr("fill", d => d.clr);

    dotg.append("text")
        .attr("x", (width / 2))
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")
        .attr('fill', 'black')
        .style("font-size", "16px")
        .style("text-decoration", "underline")
        .text(origDots[0].c_name);

    //histograms
    drawHist(origDots, svg2, "x");
    drawHist(origDots, svg2, "y");
}

//drawing histograms for the small views
function drawHist(dots, svg, or) {
    var width = 200;
    var height = 50;
    var margin = { left: 60, right: 60, top: 30, bottom: 60 }

    var totalHeight = 250 + margin.top;
    var data = [];
    if (or == "x") {
        dots.forEach(e => data.push(e.x));
    } else {
        dots.forEach(e => data.push(e.y));
    }

    var x = d3.scaleLinear()

    if (or == "x") {
        x.domain([d3.min(data), d3.max(data)])
            .range([0, width]);
    } else {
        x
            .domain([d3.min(data), d3.max(data)])
            .range([0, width]);
    }

    // set the parameters for the histogram
    var histogram = d3.histogram()
        .domain(x.domain())
        .thresholds(x.ticks(40));

    // And apply this function to data to get the bins
    var bins = histogram(data);

    var y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(bins, function (d) { return d.length; })]);   // d3.hist has to be called before the Y axis obviously

    var colors = d3.scaleLinear()
        .domain([0, d3.max(bins, function (d) { return d.length; })])
        .range([d3.rgb("steelblue").brighter(), d3.rgb("steelblue").darker()]);

    // append the bar rectangles to the svg element
    var bar = svg.append("g");
    if (or == "x") {
        bar
            .attr("transform", "translate(50,240)");
    } else {
        bar
            .attr("transform", "translate(0,240) rotate(-90) ");
    }

    bar.selectAll("rect")
        .data(bins)
        .join("rect")
        .attr("class", "rect")
        .attr("fill", d => colors(d.length))
        .attr("x", d => x(d.x0) + 1)
        .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
        .attr("y", d => y(d.length))
        .attr("height", d => y(0) - y(d.length))

    //brushing on histograms
    const brush = d3.brushX()
        .extent([[0, 0], [width, 50]])
        .on("brush", brushed)
        .on("end", brushended);

    const defaultSelection = [50, 150];

    const gb = bar
        .call(brush)
        .call(brush.move, defaultSelection);

    function brushed() {
        if (d3.event.selection) {
            svg.property("value", d3.event.selection.map(x.invert, x));
            svg.dispatch("input");

            const [x0, x1] = d3.event.selection;
            console.log(svg.select("g.scatter").selectAll("circle"));

            if (or == "x") {
                var selected = svg.select("g.scatter").selectAll("circle.y_selected");
                //var notselected = svg.select("g.scatter").selectAll("circle.not_selected");
                selected.each(function (d, i) {
                    c = d.c;
                    //console.log(d.className.baseVal);
                    if (d.x >= x.invert(x0) && d.x <= x.invert(x1)) {
                        d3.select(this)
                            //.classed("selected", true)
                            .classed("x_selected", true)
                            .attr("fill", d.clr);
                    } else {
                        d3.select(this)
                            .classed("x_selected", false)
                            .attr("fill", "grey");
                    }
                });

                updateScatter(svg.select("g.scatter").selectAll("circle.x_selected.y_selected"), svg.attr("class"));

            } else {
                var selected = svg.select("g.scatter").selectAll("circle.x_selected");

                selected.each(function (d, i) {
                    if (d.y >= x.invert(x0) && d.y <= x.invert(x1)) {
                        d3.select(this)
                            .classed("y_selected", true)
                            .attr("fill", d.clr);
                    } else {
                        d3.select(this)
                            .classed("y_selected", false)
                            .attr("fill", 'grey');
                    }
                });
                updateScatter(svg.select("g.scatter").selectAll("circle.x_selected.y_selected"), svg.attr("class"));
            }
            console.log(x.invert(x0) + " " + x.invert(x1));
            console.log(dots);
        }
    }

    function brushended() {
        if (!d3.event.selection) {
            gb.call(brush.move, defaultSelection);
        }
    }
}
