/*
 PlacementViewer loads a placement file and a list of corresponding morphologies names,
 build the morphology meshes and place them in the scene.
*/
'use strict';
var placementViewer = placementViewer ? placementViewer : {};
(function() {
  /*
   scene is where the meshes are going the be added.
   callbackRendering is the function that will be invoked after a mesh has been added.
  */
  placementViewer.PlacementViewer = function(scene, callbackRendering, callbackViewPoint) {
    this.scene = scene;
    this.callbackRendering = callbackRendering;
    this.callbackViewPoint = callbackViewPoint;
    this.averagePoint = new THREE.Vector3(0, 0, 0);
    this.count = 0;
  };
  //TODO: only loadPlacement should be public.
  placementViewer.PlacementViewer.prototype.loadPlacement = function(url) {
    // here we assume that there a .txt file containing the morphologies names
    // along with the placement file.
    return Promise.all([viewerUtils.getFile(url, 'arraybuffer'),
        viewerUtils.getFile(url.replace('.placement', '.txt'), 'text')])
        .then(function(loadedData) {
          var fData = new Float32Array(loadedData[0]);
          this.buildMorphologies(fData, loadedData[1]);
        }.bind(this));
  };

  /*
  buildMorphologies uses a pool of workers to parse the morphology files.
  These workers call loadAndPlaceMesh once they are done.
  */
  placementViewer.PlacementViewer.prototype.buildMorphologies = function(placementData, morphData) {
    var workerpool = [];
    //TODO: this should be retrieved for instance from a query parameter
    var NB_WORKERS = 7;
    for (var k = 1; k <= NB_WORKERS; k++){
      var worker = new Worker('js/parserWorker.js');
      worker.addEventListener('message', this.loadAndPlaceMesh.bind(this,placementData), false);
      workerpool.push(worker);
    }

    var morphs = morphData.split('\n');

    var rowLength = 5 * 3;
    var count = placementData.length / rowLength;

    //TODO: this should be retrieved for instance from a query parameter
    var MAX_MORPHOLOGIES = 300;
    var sampleIndices = [];
    for (var k = 0; k < count; k++){
      sampleIndices[k] = k;
    }
    var sampleIndices = _.sample(sampleIndices, Math.min(count, MAX_MORPHOLOGIES));
    for (k = 0; k < sampleIndices.length; k++){
      var morphName = morphs[sampleIndices[k]];
      //sometimes no morphologies are provided for a given soma position.
      if (morphName !== 'nan') {
        var offset = sampleIndices[k] * rowLength;
        var message = {
          'morphology': morphName,
          'offset': offset
        };
        workerpool[k % NB_WORKERS].postMessage(message);
      }
    }
  };

  /*
  loadAndPlaceMesh is the callback that will invoke the mesh creation based on the parsing of the
  morphology file and then apply the rotation and positioning from the placement file.
  */
  placementViewer.PlacementViewer.prototype.loadAndPlaceMesh = function(data, e) {
    var properties = getPlacementProperties(data, e.data.offset);

    var material = new THREE.MeshLambertMaterial( {
      color: properties.color,
      ambient: properties.color,
    } );

    var mesh = morphBuilder.buildMesh(e.data.asc, material);
    mesh.applyMatrix(properties.rotation);
    mesh.position.copy(properties.position);
    this.scene.add(mesh);
    this.averagePoint.add(properties.position);

    this.count += 1;

    var center = new THREE.Vector3();
    center.copy(this.averagePoint);
    center.divideScalar(this.count);
    this.callbackViewPoint(center);
    this.callbackRendering();
  };

  function getPlacementProperties(data, offset) {
    var p = new THREE.Vector3(data[offset], data[offset + 1], data[offset + 2]);
    offset += 3;

    var n11 = data[offset];
    var n12 = data[offset + 1];
    var n13 = data[offset + 2];
    var n14 = 0;
    offset += 3;

    var n21 = data[offset];
    var n22 = data[offset + 1];
    var n23 = data[offset + 2];
    var n24 = 0;
    offset += 3;

    var n31 = data[offset];
    var n32 = data[offset + 1];
    var n33 = data[offset + 2];
    var n34 = 0;
    offset += 3;

    var n41 = 0;
    var n42 = 0;
    var n43 = 0;
    var n44 = 1;

    var col = new THREE.Color(data[offset], data[offset + 1], data[offset + 2]);
    offset += 3;

    var m = new THREE.Matrix4();
    m.set(n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44);

    return {
      color: col,
      position: p,
      rotation: m
    };
  };
}());
