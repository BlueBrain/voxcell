'use strict';
var brainBuilderViewer = brainBuilderViewer ? brainBuilderViewer : {};

(function() {
  var defaultParticleSize = 100;
  var renderer, scene, camera, stats, controls, root;
  var cloudMaterial;
  // map of all objects loaded in the scene.
  var loadedObjects = {};

  // contains all datguiSettings
  // _datgui property contains the datgui container for that level
  // since the children folder cannot be retrieved from the root object with datgui API.
  var datguiSettings = {};

  brainBuilderViewer.main = function() {
    if (!Detector.webgl) {
      Detector.addGetWebGLMessage();
    }

    initScene();
    datguiSettings._datgui = new dat.GUI();
    datguiSettings.opacity = {};
    datguiSettings.opacity._datgui = datguiSettings._datgui.addFolder('opacity');

    datguiSettings.size = Math.log(defaultParticleSize  + 1);

    var sizeGui = datguiSettings._datgui.add(
      datguiSettings, 'size', 0.0, Math.log(5000.0)).step(0.01);

    sizeGui.onChange(function(value) {
      particleSizeChange(Math.exp(value) - 1);
    });

    if (window.location.hash) {
      loadUrl(window.location.hash.slice(1));
    }

    window.onhashchange = function() {
      loadUrl(window.location.hash.slice(1));
    };
    window.addEventListener('resize', onResize, false);
  };

  function initScene() {
    var container = document.getElementById('container');
    scene = new THREE.Scene();

    var near = 0.1;
    var far = 50000;
    scene.fog = new THREE.Fog(0x000000, near, far);
    var light = new THREE.AmbientLight(0x888888);
    scene.add(light);
    var directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    scene.add(directionalLight);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, near, far);
    camera.position.z = 150;

    controls = new THREE.TrackballControls(camera, container);
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.noZoom = false;
    controls.noPan = false;
    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.3;
    controls.keys = [65, 83, 68];
    controls.addEventListener('change', render);

    var axisHelper = new THREE.AxisHelper(5);
    scene.add(axisHelper);

    root = new THREE.Object3D();
    scene.add(root);

    renderer = new THREE.WebGLRenderer({
      antialias: false
    });
    renderer.setClearColor(scene.fog.color, 1);
    renderer.setSize(window.innerWidth, window.innerHeight);

    container.appendChild(renderer.domElement);

    stats = new Stats();
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.top = '0px';
    container.appendChild(stats.domElement);
    animate();

    controls.handleResize();
  }

  function clearRoot() {
    while (root.children.length > 0) {
      var object = root.children[0];
      object.parent.remove(object);
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
  }

  function render() {
    renderer.render(scene, camera);
    stats.update();
  }

  function updateControls(center){
    controls.target.copy(center);
    controls.update();
  }

  function loadUrl(url) {

    function addToScene(url, o) {

      function addOpacitySetting(url){
        if (datguiSettings.opacity[url]){
          return;
        }

        datguiSettings.opacity[url] = true;
        var c = datguiSettings.opacity._datgui.add(datguiSettings.opacity, url);
        c.onFinishChange(function(value) {
          setShow(url, value);
        });
      }

      function addObjectToScene(url, o){
        if (loadedObjects[url]){
          root.remove(loadedObjects[url].object);
        }
        root.add(o.object);
        updateControls(o.center);
        render();
        loadedObjects[url] = o;
      }

      addOpacitySetting(url);
      addObjectToScene(url, o);

    }

    if (url.endsWith('.mhd')) {
      loadMetaIO(url).then(addToScene.bind(null, url));
    } else if (url.endsWith('.pts')) {
      viewerUtils.getFile(url, 'arraybuffer')
      .then(buildPointCloud)
      .then(addToScene.bind(null, url));
    } else if (url.endsWith('.vcf')) {
      viewerUtils.getFile(url, 'arraybuffer')
      .then(buildVectorField)
      .then(addToScene.bind(null, url));
    } else if (url.endsWith('.placement')) {
      new placementViewer.PlacementViewer(scene, render, updateControls).loadPlacement(url);
    } else {
      console.warn('unknown extension: ' + url);
    }
  }

  function setShow(url, show){
    var o = loadedObjects[url];
    if (o && o.object){
      var mat = o.object.material;
      mat.opacity = show ? 1 : 0;
      render();
    }
  }

  function loadMetaIO(urlMhd) {
    return viewerUtils.getFile(urlMhd, 'text').then(function(data) {
      var lines = data.split('\n').map(function(line) {
        return line.split('=').map(function(s) {
          return s.trim();
        });
      });

      var mhd = {};
      lines.forEach(function(pair) {

        var NUMERICAL_KEYS = [
          'CenterOfRotation', 'DimSize', 'NDims', 'ElementSpacing',
          'Offset', 'TransformMatrix'
        ];

        if (NUMERICAL_KEYS.indexOf(pair[0]) >= 0) {
          mhd[pair[0]] = pair[1].split(' ').map(Number);
        } else {
          mhd[pair[0]] = pair[1];
        }
      });

      var urlRaw = (urlMhd.substring(0, urlMhd.lastIndexOf('/') + 1) +
        mhd.ElementDataFile);

      return viewerUtils.getFile(urlRaw, 'arraybuffer').then(buildRaw(mhd, undefined, 1, 0));
    });
  }

  function getAutoDownsampleStep(data, filterMin, maxCap) {
    var validCount = _.filter(data, function (v) { return v > filterMin; }).length;
    var downsampleStep = Math.max(Math.floor(validCount / maxCap), 1);

    console.log(
      'total data: ' + data.length +
      ' valid: ' + validCount +
      ' downsampleStep: ' + downsampleStep
    );

    return downsampleStep;
  }

  function buildRaw(mhd, downsampleStep, scaleFactor, filterMin) {
    return function(data) {
      var METAIO_TYPE_MAP = {
        'MET_UCHAR': Uint8Array,
        'MET_UINT': Uint32Array,
        'MET_FLOAT': Float32Array,
      };

      var data = new METAIO_TYPE_MAP[mhd.ElementType](data);

      var dimsizeX = mhd.DimSize[0];
      var dimsizeY = mhd.DimSize[1];
      var dimsizeZ = mhd.DimSize[2];

      var geometry = new THREE.Geometry();
      var maxValue = _.max(data);
      var minValue = Math.max(filterMin, _.min(data));
      var count = 0;

      var getValue = function(x, y, z) {
        // Fortran order
        var val = data[z * (dimsizeX * dimsizeY) + y * dimsizeX + x];
        return val;
      };

      var scale = new THREE.Vector3(
        mhd.ElementSpacing[0] * scaleFactor,
        mhd.ElementSpacing[1] * scaleFactor,
        mhd.ElementSpacing[2] * scaleFactor
      );

      var averagePoint = new THREE.Vector3(0, 0, 0);

      if (downsampleStep === undefined) {
        downsampleStep = getAutoDownsampleStep(data, filterMin, 100000);
      }

      var i = 0;
      for (var z = 0; z < dimsizeZ; z++) {
        for (var y = 0; y < dimsizeY; y++) {
          for (var x = 0; x < dimsizeX; ++x) {
            i++;
            if ((i % downsampleStep) == 0) {

              var value = getValue(x, y, z);
              if (value > filterMin) {

                var p = new THREE.Vector3(x, y, z).multiply(scale);
                geometry.vertices.push(p);

                var intensity = (value - minValue) / (maxValue - minValue);
                geometry.colors.push(
                  new THREE.Color(intensity, 0, 1 - intensity)
                );

                averagePoint.add(p);
                count++;
              }
            }
          }
        }
      }

      console.log('max: ' + maxValue + ' min: ' + minValue);
      console.log('loaded: ' + count + ' points');

      cloudMaterial = buildSquareCloudMaterial(defaultParticleSize);

      return {
        object: new THREE.PointCloud(geometry, cloudMaterial),
        center: averagePoint.divideScalar(count)
      };
    };
  }

  function buildPointCloud(data) {
    var data = new Float32Array(data);
    var rowLength = 2 * 3; // point and color (3 components each)
    var count = data.length / rowLength;
    var geometry = new THREE.Geometry();
    var averagePoint = new THREE.Vector3(0, 0, 0);

    for (var i = 0; i < count; ++i) {
      var offset = i * rowLength;
      var x = data[offset];
      var y = data[offset + 1];
      var z = data[offset + 2];

      offset = i * rowLength + 3;
      var r = data[offset];
      var g = data[offset + 1];
      var b = data[offset + 2];

      var p = new THREE.Vector3(x, y, z);
      averagePoint.add(p);

      geometry.vertices.push(p);
      geometry.colors.push(new THREE.Color(r, g, b));
    }

    console.log('loaded: ' + count + ' points');

    cloudMaterial = buildCircleCloudMaterial(defaultParticleSize);

    return {
      object: new THREE.PointCloud(geometry, cloudMaterial),
      center: averagePoint.divideScalar(count)
    };
  }

  function buildVectorField(data) {
    var data = new Float32Array(data);
    var rowLength = 4 * 3; // p0, color0, p1, color1 (3 components each)
    var count = data.length / rowLength;
    var geometry = new THREE.Geometry();
    var averagePoint = new THREE.Vector3(0, 0, 0);
    var scale = 1.0;
    var lineLength = 100;

    for (var i = 0; i < count; ++i) {

      var offset = i * rowLength;
      var x0 = data[offset] * scale;
      var y0 = data[offset + 1] * scale;
      var z0 = data[offset + 2] * scale;

      var p0 = new THREE.Vector3(x0, y0, z0);
      averagePoint.add(p0);
      geometry.vertices.push(p0);

      offset = i * rowLength + 3;
      var r0 = data[offset];
      var g0 = data[offset + 1];
      var b0 = data[offset + 2];

      geometry.colors.push(new THREE.Color(r0, g0, b0));

      offset = i * rowLength + 6;
      var x1 = x0 + data[offset] * lineLength;
      var y1 = y0 + data[offset + 1] * lineLength;
      var z1 = z0 + data[offset + 2] * lineLength;

      geometry.vertices.push(new THREE.Vector3(x1, y1, z1));

      offset = i * rowLength + 9;
      var r1 = data[offset];
      var g1 = data[offset + 1];
      var b1 = data[offset + 2];

      geometry.colors.push(new THREE.Color(r1, g1, b1));
    }

    console.log('loaded: ' + count + ' lines');

    var material = new THREE.LineBasicMaterial({
      linewidth: 2,
      color: 0xffffff,
      vertexColors: THREE.VertexColors
    });

    return {
      object: new THREE.Line(geometry, material, THREE.LinePieces),
      center: averagePoint.divideScalar(count)
    };
  }

  function buildWireFrameCuboid(x, y, z) {
    var cuboid = new THREE.Object3D();
    var dims = new THREE.Vector3(x / 2, y / 2, z / 2);

    var material = new THREE.LineBasicMaterial({
      color: 0x000000
    });

    var geometry = new THREE.Geometry();
    geometry.vertices.push(new THREE.Vector3(dims.x, dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, -dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, -dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, dims.y, dims.z));

    geometry.vertices.push(new THREE.Vector3(dims.x, dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, -dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, -dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, dims.y, -dims.z));
    var line = new THREE.Line(geometry, material);
    cuboid.add(line);

    var geometry = new THREE.Geometry();
    geometry.vertices.push(new THREE.Vector3(-dims.x, dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, -dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(-dims.x, -dims.y, -dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, -dims.y, dims.z));
    geometry.vertices.push(new THREE.Vector3(dims.x, -dims.y, -dims.z));
    var line = new THREE.Line(geometry, material, THREE.LinePieces);
    cuboid.add(line);
    return cuboid;
  }

  function buildSquareCloudMaterial(size) {
    return new THREE.PointCloudMaterial({
      size: size,
      vertexColors: THREE.VertexColors,
      transparent: true,
      opacity: 1
    });
  }

  function buildCircleCloudMaterial(size) {
    var c = $('<canvas width="256" height="256" />').get(0);
    var ctx = c.getContext('2d');
    ctx.beginPath();
    ctx.arc(c.width >> 1, c.height >> 1, c.width >> 1, 2 * Math.PI, false);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    var tex = new THREE.Texture(c);
    tex.needsUpdate = true;

    return new THREE.PointCloudMaterial({
      size: size,
      vertexColors: THREE.VertexColors,
      transparent: true,
      opacity: 1,
      alphaTest: 0.5,
      map: tex
    });
  }

  function particleSizeChange(amount) {
    cloudMaterial.size = amount;
    console.log('cloudMaterial size', cloudMaterial.size);
    render();
  }

  function onResize(evt) {
    var width = window.innerWidth;
    var height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    controls.handleResize();
    render();
  }
}());
