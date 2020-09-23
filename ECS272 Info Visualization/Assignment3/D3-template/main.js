(function () {
  // first, load the dataset from a CSV file
  d3.csv("https://data.sfgov.org/api/views/xzie-ixjw/rows.csv?accessType=DOWNLOAD")
    .then(csv => {
      // log csv in browser console
      console.log(csv);

      // create data by selecting two columns from csv 
      var data = csv.map(row => {
        return {
          yes: Number(row['Yes Votes']),
          no: Number(row['No Votes']),
          subject: String(row['Subject'])
        }
      })

      /********************************* 
      * Visualization codes start here
      * ********************************/
      var width = 600;
      var height = 400;
      var margin = {left: 60, right: 20, top: 20, bottom: 60}

      var svg = d3.select('#container')
        .append('svg')
          .attr('width', width + margin.left + margin.right)
          .attr('height', height + margin.top + margin.bottom) 

      var view = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      //scale functions
      var x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.yes)])
        .range([0, width]);
        
      var y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.no)])
        .range([height, 0]);

      
      // create a scatter plot
      var scatterPlot = view.selectAll('circle')
        .data(data)
        .enter()
          .append('circle')
          .attr('cx', d => x(d.yes))
          .attr('cy', d => y(d.no))
          .attr('data-x', d => d.yes)
          .attr('data-y', d => d.no)
          .attr("r", 8)
          .attr('opacity', 0.5)
          .attr("fill", "orange")
          console.log(scatterPlot);
          var lasso_start = function () {
            console.log('start')
            lasso.items()
                .attr("r", 7)
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
                .attr("r", 7);
            lasso.notSelectedItems()
                .attr("r", 5);
        };
        //console.log(circles[0]);
        //var s = d3.select("#scatter > svg");
        //var cir = d3.select("#scatter > svg").selectAll("circle");
        const lasso = d3.lasso()
            .closePathDistance(305)
            .closePathSelect(true)
            .area(svg)
            .items(scatterPlot)
            .on("start", lasso_start)
            .on("draw", lasso_draw)
            .on("end", lasso_end);
        svg.call(lasso);

      var tooltip = document.getElementById('tooltip')
      scatterPlot
        .on('mouseenter', function(d) {
          d3.select(this).style('fill', 'blue')
          tooltip.innerHTML = 'Yes Votes = ' + d.yes + ', No Votes = ' + d.no + ', Subject = ' + d.subject
        })
        .on('mouseleave', function(d) {
          d3.select(this).style('fill', 'orange')
        })

      // x axis
      view.append("g")	
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).ticks(6))
      	.append("text")
          .attr("fill", "#000")
          .attr("x", width / 2)
          .attr('y', margin.bottom / 2)
          .attr("dy", "0.71em")
          .attr("text-anchor", "end")
          .text("Yes Votes");

      // y axis
      view.append("g")
        .call(d3.axisLeft(y).ticks(6))
        .append("text")
          .attr("fill", "#000")
          .attr("transform", "rotate(-90)")
          .attr("x", - height / 2)
          .attr("y", - margin.left)
          .attr("dy", "0.71em")
          .attr("text-anchor", "end")
          .text("No Votes");

    })
})()