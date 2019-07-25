import Map from 'ol/Map.js';
import View from 'ol/View.js';
import {getWidth, getTopLeft} from 'ol/extent.js';
import TileLayer from 'ol/layer/Tile.js';
import {get as getProjection} from 'ol/proj.js';
import OSM from 'ol/source/OSM.js';
import WMTS from 'ol/source/WMTS.js';
import WMTSTileGrid from 'ol/tilegrid/WMTS.js';


var projection = getProjection('EPSG:3857');
var projectionExtent = projection.getExtent();
var size = getWidth(projectionExtent) / 512;
var resolutions = new Array(14);
var matrixIds = new Array(14);
for (var z = 0; z < 14; ++z) {
// generate resolutions and matrixIds arrays for this WMTS
resolutions[z] = size / Math.pow(2, z);
matrixIds[z] = z;

console.log(resolutions[z], matrixIds[z], size);
}

var map = new Map({
layers: [
  new TileLayer({
    opacity: 1.0,
    source: new WMTS({
      attributions: 'example data',
      url: 'http://localhost:5000/wmts',
      layer: '0',
      matrixSet: 'EPSG:3857',
      format: 'image/png',
      projection: projection,
      tileGrid: new WMTSTileGrid({
	origin: getTopLeft(projectionExtent),
	resolutions: resolutions,
	matrixIds: matrixIds
      }),
      style: 'default',
      wrapX: false
    })
  })
],
target: 'map',
view: new View({
  center: [-11158582, 15013697],
  zoom: 3
})
});
