import Map from 'ol/Map.js';
import View from 'ol/View.js';
import {getWidth, getTopLeft} from 'ol/extent.js';
import TileLayer from 'ol/layer/Tile.js';
import {get as getProjection} from 'ol/proj.js';
import OSM from 'ol/source/OSM.js';
import WMTS from 'ol/source/WMTS.js';
import WMTSTileGrid from 'ol/tilegrid/WMTS.js';
import Draw, {createRegularPolygon, createBox} from 'ol/interaction/Draw.js';
import {Vector as VectorSource} from 'ol/source.js';


var projection = getProjection('EPSG:4326');
var projectionExtent = projection.getExtent();
var size = getWidth(projectionExtent) / 512;
var resolutions = new Array(14);
var matrixIds = new Array(14);
for (var z = 0; z < 14; ++z) {
// generate resolutions and matrixIds arrays for this WMTS
resolutions[z] = size / Math.pow(2, z);
matrixIds[z] =  z;

// resolutions[0] = resolutions[0] / 2;

console.log(projectionExtent, resolutions[z], matrixIds[z], size);
}

var url = new URL('http://localhost:5000/wmts')

if ($('#cpu')[0].checked) {
  url.searchParams.append('device', 'cpu');
}

if ($('#cmap')[0].checked) {
  if ($('#random')[0].checked) {
    url.searchParams.append('cmap', 'random');
  } else {
    url.searchParams.append('cmap', 'VIIRS_SNPP_Brightness_Temp_BandI5_Day');
  }
} else {
  if ($('#random')[0].checked) {
    url.searchParams.append('cmap', 'random');
  }
}

if ($('#filter')[0].checked) {
  url.searchParams.append('filter', 'sobel')
}

if ($('#scale')[0].checked) {
  url.searchParams.append('scale', 'true')
  url.searchParams.append('min_value', '10.0')
  url.searchParams.append('max_value', '30.0')
}

console.log(getTopLeft(projectionExtent));

var map = new Map({
layers: [
  // new TileLayer({
  //   source: new OSM(),
  //   opacity: 0.7,
  //   projection: "EPSG:4326",
  //   style: 'default',
  //   wrapX: false
  // }),

  new TileLayer({
    opacity: 1.0,
    source: new WMTS({
      attributions: 'example data',
      url: url.toString(),
      layer: '0',
      matrixSet: 'EPSG:4326',
      format: 'image/png',
      projection: projection,
      tileGrid: new WMTSTileGrid({
	origin: getTopLeft(projectionExtent),
	resolutions: resolutions,
  matrixIds: matrixIds,
  tileSize: 512
      }),
      style: 'default',
      wrapX: false
    }),
  })
],
target: 'map',
view: new View({
  center: [0, 0],
  zoom: 2,
  // displayProjection: 'EPSG:4326',
  projection: 'EPSG:4326',
})
});

map.on('moveend',function(e){
  // let extent = e.feature.getGeometry().getExtent();
  let extent = map.getView().calculateExtent(map.getSize());
  fetch("http://localhost:5000/getstats?bbox=" + extent.join(",")).then(
      response => response.json()).then(function(data) {
        $('#stats').text("Max: " + data.max.toString() + 
      ", Min: " + data.min.toString() + ", Mean: " + data.mean.toString() + ", Std: " + 
      data.std.toString()); 
      $('#bbox').text("[" + extent.join(",") + "]");
      });
});

var source = new VectorSource({wrapX: false});

var draw; // global so we can remove it later
function addInteraction() {
    var value = 'Circle';
    var geometryFunction = createBox();
    draw = new Draw({
        source: source,
        type: value,
        geometryFunction: geometryFunction,
        projection: "EPSG:4326"
        });

    draw.on('drawend',function(e){
            // let extent = e.feature.getGeometry().getExtent();
            let extent = e.feature.getGeometry();
            console.log(extent);
            extent = extent.extent_;
            fetch("http://localhost:5000/getstats?bbox=" + extent.join(",")).then(response => response.json()).then(data => $('#stats').text("Max: " + data.max.toString() + ", Min: " + data.min.toString() + ", Mean: " + data.mean.toString() + ", Std: " + data.std.toString()));
    });
    map.addInteraction(draw);
}

addInteraction();
