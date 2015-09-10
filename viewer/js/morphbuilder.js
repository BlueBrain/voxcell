/*
 morph builder takes a list of tree structures and a THREE material and
 builds a mesh out of it.
 the tree structure is just an array of composed of:
  - array of structure.
  - point. An object with 'x', 'y', 'z', 'd' that are floats.
*/
'use strict';
var morphBuilder = morphBuilder ? morphBuilder : {};

(function() {
  morphBuilder.buildMesh = function(data, material) {
    var g = new THREE.BufferGeometry();
    var positionArray = [];
    var indexArray = [];

    var attributes = g.attributes;
    var curOffset = {
      start: 0,
      count: 0,
      index: 0
    };
    g.offsets = [curOffset];
    var indices = {
      posOfs: 0,
      indexOfs: 0,
      curOffset: curOffset,
      vertexCount: 0
    };
    for (var idx = 0; idx < data.length; idx++){
      buildSection(data[idx].data, g,indices, positionArray, indexArray);
    }

    var mesh = createMesh(g, positionArray, indexArray, material);
    return mesh;
  };

  function createMesh(g, positionArray, indexArray, material){
    var uArray = new Uint16Array(indexArray.length);
    uArray.set(indexArray);

    g.attributes.index = {
      itemSize: 1,
      array: uArray,
      numItems: uArray.length
    };
    var fArray = new Float32Array(positionArray.length);
    fArray.set(positionArray);

    g.attributes.position = {
      itemSize: 3,
      array: fArray,
      numItems: fArray.length
    };
    g.computeVertexNormals();
    var mesh = new THREE.Mesh(g, material);
    return mesh;
  }

  function addVertice(parentPt, pt, indices, positionArray, indexArray){
    var startPoint = new THREE.Vector3(parentPt.x,
       parentPt.y,
       parentPt.z);
    var endPoint = new THREE.Vector3(pt.x,
        pt.y,
        pt.z);
    var startRadius = parentPt.d;
    var endRadius = pt.d;
    var zAxis = new THREE.Vector3(0, 0, 1);
    var localZAxis = new THREE.Vector3(endPoint.x - startPoint.x,
        endPoint.y - startPoint.y,
        endPoint.z - startPoint.z);
    var q = new THREE.Quaternion();
    var t = new THREE.Vector3().crossVectors(zAxis, localZAxis);
    q.x = t.x;
    q.y = t.y;
    q.z = t.z;
    q.w = Math.sqrt((zAxis.lengthSq() * localZAxis.lengthSq())) + zAxis.dot(localZAxis);
    q.normalize();

    //TODO: this should be an input parameter.
    var nbVerticesBase = 3;

    addBase(startPoint,q, startRadius, positionArray, indices, nbVerticesBase);
    addBase(endPoint, q, endRadius, positionArray, indices, nbVerticesBase);
    addTriangles(indexArray, indices, nbVerticesBase);
  }

  function addTriangles(indexArray, indices, nbVerticesBase){
    var startIdx = indices.posOfs / 3  - 2 * nbVerticesBase;

    //build 2 triangles that compose the face of the rectangle made by
    //2 vertices of the base of the cylinder (from startIdx) baseA and baseB
    //2 vertices of the top of the cylinder (from end_idx) topA and topB
    for (var n = 0; n < nbVerticesBase; n++){
      var baseA = startIdx + (n) % nbVerticesBase;
      var baseB = startIdx + (n + 1) % nbVerticesBase;
      var topA = startIdx + nbVerticesBase + (n) % nbVerticesBase;
      var topB = startIdx + nbVerticesBase + (n + 1) % nbVerticesBase;
      //triangle 1
      indexArray[indices.indexOfs] = topA - indices.curOffset.index;
      indices.indexOfs++;
      indexArray[indices.indexOfs] = baseB - indices.curOffset.index;
      indices.indexOfs++;
      indexArray[indices.indexOfs] = baseA - indices.curOffset.index;
      indices.indexOfs++;
      //triangle 2
      indexArray[indices.indexOfs] = baseB - indices.curOffset.index;
      indices.indexOfs++;
      indexArray[indices.indexOfs] = topA - indices.curOffset.index;
      indices.indexOfs++;
      indexArray[indices.indexOfs] = topB - indices.curOffset.index;
      indices.indexOfs++;
      indices.curOffset.count += 6;
    }
  }

  function addBase(center, q, radius, positionArray, indices, nbVerticesBase) {
    for (var n = 0 ; n < nbVerticesBase; n++){
      var phi = (n * Math.PI * 2) / nbVerticesBase;
      var x = Math.cos(phi) * radius;
      var y = Math.sin(phi) * radius;
      var v = new THREE.Vector3(x, y, 0);
      v.applyQuaternion(q);
      v.add(center);
      positionArray[indices.posOfs] = v.x;
      indices.posOfs ++;
      positionArray[indices.posOfs] = v.y;
      indices.posOfs ++;
      positionArray[indices.posOfs] = v.z;
      indices.posOfs ++;
      indices.vertexCount ++;
    }
  }
  function distance(p1, p2){
    var v1 = new THREE.Vector3(p1.x,p1.y,p1.z);
    var v2 = new THREE.Vector3(p2.x,p2.y,p2.z);
    return v1.distanceTo(v2);
  }
  /*
  buildSection builds a simplified version of the morphology:
  it ignores intermediate section points.
  it ignores leaf sections that are small compared to the distance from the soma.
  */
  function buildSection(sectionData, g, indices, positionArray, indexArray){
    var curParent;
    var curPoint;
    for (var idx = 0; idx < sectionData.length; idx++){
      var curItem = sectionData[idx];
      if (indices.vertexCount > 50000){
        indices.curOffset = {
          start: indices.posOfs,
          count: 0,
          index: indices.posOfs / 3
        };
        g.offsets.push(indices.curOffset);
        indices.vertexCount = 0;
      }
      if (curItem instanceof Array){
        if (curPoint && curParent){
          addVertice(curParent, curPoint, indices, positionArray, indexArray);
        }
        buildSection(curItem,g, indices, positionArray, indexArray);
        curParent = undefined;
        curPoint = undefined;
        continue;
      }

      if (curItem instanceof Object){
        if (curParent === undefined) {
          curParent = curItem;
          continue;
        }
        curPoint = curItem;
      }
    }
    if (curPoint && curParent){
      // this removes leaf section too small compare to the distance to the soma.
      if (distance(curPoint,curParent) * 5 > distance(curParent,{'x': 0,'y': 0,'z': 0})){
        addVertice(curParent, curPoint, indices, positionArray, indexArray);
      }
    }
  }
}());
