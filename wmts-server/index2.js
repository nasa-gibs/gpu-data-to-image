import Map from 'ol/Map.js';
import View from 'ol/View.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import TileLayer from 'ol/layer/Tile.js';
import OSM from 'ol/source/OSM.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';

/** get value of currently selected option */
function getValue(selector) {
    return selector.options[selector.selectedIndex].value;
}

/** get name of a given layer, can be null */
function getName(layer) {
    return layer.values_.source.layer_;
}

/* find layer by name in a given map */
function findByName(map, name) {
    let layers = map.getLayers();
    let length = layers.getLength();

    console.log("finding layer " + name + " in map with " + length + " layers.");

    for (let i = 0; i < length; i++) {
        let current = getName(layers.item(i));
        if (name === current) {
            return layers.item(i);
        }
    }
    
    return null;
}

function makeLayer(result, layer_name, date, method) {
        let options = optionsFromCapabilities(result, {
            layer: layer_name
        });

        options.wrapX = false;
        options.dimensions.Time = date;
        // options.urls[0] = options.urls[0] // remove the leading http:// (temporarily)
        //    .replace(/\/+/g, '/')       // replace consecutive slashes with a single slash
        //    .replace(/\/+$/, '');       // remove trailing slashes


        console.log(options);
        let layer = new TileLayer({
              opacity: 1,
              source: new WMTS(/** @type {!module:ol/source/WMTS~Options} */ (options)),
        });

        return layer;
}

function createWMTSMap(url, default_layer, date) {
    let parser = new WMTSCapabilities();

    fetch(url).then(function(response) {
        return response.text();
    }).then(function(text) {
        /*console.log(text.replace(/onearth-tile-services/g, "localhost"));*/
        let result = parser.read(text.replace(/onearth-tile-services/g, "localhost/"));
        console.log(result);
        let layer = makeLayer(result, default_layer, date, "none");        

        let map = new Map({
          layers: [ layer ],
          target: 'map',
          view: new View({
            center: [0.0, 0.0], /**[19412406.33, -5050500.21],*/
            zoom: 2
          })
        });

    
        /* create a dropdown menu with these items */
        let div = document.querySelector("#container");

        let new_frag = document.createDocumentFragment(),
        method_select = document.createElement("select");

        for (let option of ["none", "sobel", "downsample", "blur"]) {
            method_select.options.add(new Option(option, option));
        }

        new_frag.insertBefore(method_select, new_frag.firstChild);
        div.insertBefore(new_frag, div.firstChild);


        let frag = document.createDocumentFragment(),
        select = document.createElement("select");
        select.setAttribute('id', 'selector');

        console.log(result.Contents.Layer);
        for (let option of result.Contents.Layer) {
            select.options.add(new Option(option.Title, option.Identifier));
        }

        frag.insertBefore(select, frag.firstChild);
        div.insertBefore(frag, div.firstChild);

        /* setup callback function for add and remove */
        $(document).ready(function() {
            $('#addOSM').on('click', function() {
                let new_layer = makeLayer(result, getValue(select), date, getValue(method_select));
                map.addLayer(new_layer);
            });

            $('#remOSM').on('click', function() {
                let current = findByName(map, getValue(select));
                console.log(current); 
                map.removeLayer(current);
            });
        });

    });

    return map;
}


/* this is the actual GIBS server */
let map = createWMTSMap(
    "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&request=GetCapabilities",
    'MODIS_Terra_CorrectedReflectance_TrueColor',
    '2016-06-04'
)

/* this is the localhost demo server */
/* let map = createWMTSMap(
    "http://localhost/onearth/wmts/epsg4326/wmts.cgi?SERVICE=WMTS&request=GetCapabilities",
    'blue_marble',
    '2016-06-04'
)  */


/*let map = createWMTSMap(
    "http://localhost/oe-status/wmts.cgi?service=WMTS&request=GetCapabilities&version=1.0.0",
    'BlueMarble16km',
    '2004-08-01'
) */


// let map = createWMTSMap(
//     "http://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml",
//     'MODIS_Terra_CorrectedReflectance_TrueColor',
//     '2017-01-01'
// )
