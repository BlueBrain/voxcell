'use strict';
var brainBuilderViewer = brainBuilderViewer || {};
// manage the dependency manually
var BigScreen = require('bigscreen');
(function() {
  var DEFAULTPARTICLESIZE = 25;
  var NEAR = 0.1;
  var FAR = 50000;
  var HELPERSIZE = 120;
  var AXISSIZE = 5;
  var HELPERCAMERALENGTH = 10;
  brainBuilderViewer.Viewer = function(inputContainer, helperContainer, displayParameters) {

    this.container = inputContainer;
    this.helperContainer = helperContainer;

    // map of all objects loaded in the scene.
    this.loadedObjects = {};

    // contains all datguiSettings
    // _datgui property contains the datgui container for that level
    // since the children folder cannot be retrieved from the root object with datgui API.
    this.datguiSettings = {};
    this.displayParameters = displayParameters;

    // privileged methods
    var that = this;

    this.initScene = function() {
      that.scene = new THREE.Scene();
      that.scene.fog = new THREE.Fog(0x000000, NEAR, FAR);
      var light = new THREE.AmbientLight(0x888888);
      that.scene.add(light);
      var directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
      that.scene.add(directionalLight);
      var axisHelper = new THREE.AxisHelper(AXISSIZE);
      that.scene.add(axisHelper);
      that.root = new THREE.Object3D();
      that.scene.add(that.root);
    };

    this.render = function() {
      if (that.renderer !== undefined){
        that.renderer.render(that.scene, that.camera);
      }
      if (that.rendererHelper !== undefined){
        that.rendererHelper.render(that.sceneHelper, that.cameraHelper);
      }
    };

    this.animate = function() {
      that.updateAxisHelper();

      if (that.controls){
        that.controls.update();
      }

      requestAnimationFrame(that.animate);
      that.render();
    };

    this.updateAxisHelper = function() {
      if (that.cameraHelper){
        that.cameraHelper.position.copy(that.camera.position);
        that.cameraHelper.position.sub(that.controls.target);
        that.cameraHelper.position.setLength(HELPERCAMERALENGTH);
        that.cameraHelper.lookAt(that.sceneHelper.position);
      }
    };

    this.updateControls = function(center) {
      if (that.controls){
        that.controls.target.copy(center);
        that.controls.update();
      }
    };

    this.setShow = function(url, show) {
      var o = that.loadedObjects[url];
      if (o && o.object){
        var mat = o.object.material;
        mat.opacity = show ? 1 : 0;
        that.render();
      }
    };

    this.onResize = function(evt) {
      // this forces the renderer to have the same size
      // as the containing cell and container.clientWidth
      // to be updated.
      that.renderer.domElement.style.height = '100%';
      that.renderer.domElement.style.width = '100%';

      _.debounce(function() {
        var width = that.container.clientWidth;
        var height = that.container.clientHeight;
        that.renderer.setSize(width, height);
        that.camera.aspect = width / height;
        that.camera.updateProjectionMatrix();
        that.render();
      }, 250)();

    };

    this.particleSizeChange = function(amount) {
      if (that.cloudMaterial.uniforms !== undefined){
        that.cloudMaterial.uniforms.size.value = amount;
      } else {
        that.cloudMaterial.size = amount;
      }
      that.render();
    };

    this.addSliceSettings = function() {
      var bb = that.geometry.boundingBox;
      that.datguiSettings.slice = {};
      that.datguiSettings.slice._datgui = that.datguiSettings._datgui.addFolder('slice');

      ['x', 'y', 'z'].forEach(function(value, idx) {
        that.cloudMaterial.uniforms.threshold.value[idx] = bb.min[value];
        that.datguiSettings.slice[value] = bb.min[value];

        var spliceGui = that.datguiSettings.slice._datgui
              .add(that.datguiSettings.slice,
                   value,
                   bb.min[value],
                   bb.max[value]);

        spliceGui.onChange(function(value) {
          that.cloudMaterial.uniforms.threshold.value[idx] = value;
          that.render();
        });
      });
    };

    this.addShaders = function() {
      var vertexShader = $('<script type=\'x-shader/x-vertex\' id=\'vertexshader\'>\
               uniform float size;\
               uniform vec3 threshold;\
               attribute vec3 customColor;\
               varying vec3 vColor;\
               varying float vDiscard;\
               void main() {\
                 vColor = customColor;\
                 vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\
                 gl_PointSize = size * ( 300.0 / length( mvPosition.xyz ) );\
                 gl_Position = projectionMatrix * mvPosition;\
                 if (position[0] < threshold[0] || \
                     position[1] < threshold[1] || \
                     position[2] < threshold[2]  ){\
                         vDiscard=1.0;}\
                 else {vDiscard=0.0;}\
               }\
               </script>');
      var fragmentShader = $('<script type=\'x-shader/x-fragment\' id=\'fragmentshader\'>\
               uniform vec3 color;\
               uniform sampler2D texture;\
               varying vec3 vColor;\
               varying float vDiscard;\
               void main() {\
                 gl_FragColor = vec4(color * vColor, 1.0);\
                 gl_FragColor = gl_FragColor * texture2D(texture, gl_PointCoord);\
                 if (gl_FragColor.w < 0.5 || vDiscard != 0.0){discard;}\
                }\
              </script>');
      that.container.appendChild(vertexShader[0]);
      that.container.appendChild(fragmentShader[0]);
    };

    this.initAxisHelper = function() {
      that.sceneHelper = new THREE.Scene();
      var axisSceneHelper = new THREE.AxisHelper(AXISSIZE);
      that.sceneHelper.add(axisSceneHelper);
      that.rendererHelper = new THREE.WebGLRenderer({alpha: true});
      that.rendererHelper.setClearColor(0x000000, 0);
      that.rendererHelper.setSize(HELPERSIZE, HELPERSIZE);
      that.helperContainer.appendChild(that.rendererHelper.domElement);
      that.cameraHelper = new THREE.PerspectiveCamera(30,
                                                      1,
                                                      NEAR, FAR);
      that.cameraHelper.up = that.camera.up;
      that.updateAxisHelper();
    };
  };

  brainBuilderViewer.Viewer.constructor = brainBuilderViewer.Viewer;

  brainBuilderViewer.Viewer.prototype = {

    constructor: brainBuilderViewer.Viewer,

    init: function() {
      if (!Detector.webgl) {
        Detector.addGetWebGLMessage();
      }
      this.addShaders();
      this.initScene();
      var gui = new dat.GUI({autoPlace: false});
      this.datguiSettings._datgui = gui;
      this.container.appendChild(gui.domElement);
      // place settings gui on the top right of the container.
      gui.domElement.style.position = 'absolute';
      gui.domElement.style.right = '0px';
      gui.domElement.style.top = '0px';
      var settings = this.datguiSettings;
      settings.opacity = {};
      settings.opacity._datgui = settings._datgui.addFolder('opacity');
      var paramParticleSize = parseFloat(this.displayParameters.particle_size);
      settings.size =  paramParticleSize || Math.log(DEFAULTPARTICLESIZE + 1);

      var sizeGui = settings._datgui.add(
          settings, 'size', 0.0, Math.log(5000.0)).step(0.01);

      var that = this;
      if (BigScreen.enabled) {
        settings._datgui.add(
                    {
                      fullscreen: function() {
                        BigScreen.toggle(that.container, that.onResize, that.onResize);
                      }
                    },
                    'fullscreen'
                );
      }

      settings._datgui.close();

      sizeGui.onChange(function(value) {
        that.particleSizeChange(Math.exp(value) - 1);
      });
    },

    onShow: function() {
      this.renderer = new THREE.WebGLRenderer({
        antialias: false
      });
      this.renderer.setClearColor(this.scene.fog.color, 1);

      this.renderer.domElement.style.display = 'block';
      this.renderer.domElement.style.height = '100%';
      this.renderer.domElement.style.width = '100%';
      this.container.appendChild(this.renderer.domElement);

      var height = this.container.clientHeight;
      var width = this.container.clientWidth;

      this.renderer.setSize(width, height);

      this.camera = new THREE.PerspectiveCamera(60,
                                                width / height,
                                                NEAR, FAR);
      this.camera.position.z = 150;

      this.controls = new THREE.TrackballControls(this.camera,
                                                  this.renderer.domElement);
      var controls = this.controls;
      controls.rotateSpeed = 1.0;
      controls.zoomSpeed = 1.2;
      controls.panSpeed = 0.8;
      controls.noZoom = false;
      controls.noPan = false;
      controls.staticMoving = true;
      controls.dynamicDampingFactor = 0.3;
      controls.keys = [65, 83, 68];
      controls.addEventListener('change', this.render);
      this.initAxisHelper();
      this.animate();
    },

    onRemove: function() {
      function removeElemChildren(elem){
        while (elem.lastChild) {
          elem.removeChild(elem.lastChild);
        }
      };

      delete this.renderer ;
      delete this.rendererHelper;
      delete this.cameraHelper;
      delete this.camera ;
      delete this.controls ;
      removeElemChildren(this.container);
      delete this.container ;
      removeElemChildren(this.helperContainer);
      delete this.helperContainer;
      delete this.sceneHelper;
      delete this.scene ;
      delete this.root ;

    },

    loadUrl: function(url, promise, shape, dtype) {
      var that = this;
      function addToScene(url, o) {
        function addOpacitySetting(url){
          if (that.datguiSettings.opacity[url]){
            return;
          }
          that.datguiSettings.opacity[url] = true;
          var c = that.datguiSettings.opacity._datgui.add(that.datguiSettings.opacity, url);
          c.onFinishChange(function(value) {
            that.setShow(url, value);
          });
        }

        function addObjectToScene(url, o){
          if (that.loadedObjects[url]){
            that.root.remove(that.loadedObjects[url].object);
          }
          that.root.add(o.object);
          that.updateControls(o.center);
          that.render();
          that.loadedObjects[url] = o;
        }

        addOpacitySetting(url);
        addObjectToScene(url, o);
      }

      if (!promise){
        promise = viewerUtils.getFile(url, 'arraybuffer');
      }
      if (url.endsWith('.raw')) {
        // this is temporary as we are moving to nrrd.
        promise
            .then(buildRaw({'DimSize': shape,
                            'ElementSpacing': [25,25,25],
                            'ElementType': dtype},
                           undefined, 1, 0).bind(this))
            .then(addToScene.bind(null, url));
      } else if (url.endsWith('.pts')) {
        promise
            .then(buildPointCloud.bind(this))
            .then(addToScene.bind(null, url));
      } else if (url.endsWith('.vcf')) {
        promise
            .then(buildVectorField.bind(this))
            .then(addToScene.bind(null, url));
      } else if (url.endsWith('.placement')) {
        //TODO  implement it.
        //new placementViewer.PlacementViewer(scene, render, updateControls).loadPlacement(url);
      } else {
        console.warn('unknown extension: ' + url);
      }
    }
  };

  function getAutoDownsampleStep(data, filterMin, maxCap) {
    var validCount = _.filter(data, function(v) { return v > filterMin; }).length;
    var downsampleStep = Math.max(Math.floor(validCount / maxCap), 1);

    console.log(
      'total data: ' + data.length +
      ' valid: ' + validCount +
      ' downsampleStep: ' + downsampleStep
    );

    return downsampleStep;
  }

  function buildRaw(mhd, downsampleStep, scaleFactor, filterMin) {
    return function(inputData) {
      var DTYPE_TYPE_MAP = {
        'uint8': Uint8Array,
        'uint32': Uint32Array,
        'float32': Float32Array
      };

      var data = new DTYPE_TYPE_MAP[mhd.ElementType](inputData);
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

      var palette = [
        new THREE.Color(0x0571b0),
        new THREE.Color(0x92c5de),
        new THREE.Color(0xb392de),
        new THREE.Color(0xca0020)
      ];

      var transferFunction = function(intensity, palette) {
        var idx = Math.max(0, Math.ceil(intensity * (palette.length - 1)) - 1);
        var c = new THREE.Color(palette[idx].r, palette[idx].g, palette[idx].b);
        c.lerp(palette[idx + 1], intensity);
        return c;
      };

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
                if (maxValue > minValue) {
                  var intensity = (value - minValue) / (maxValue - minValue);
                } else {
                  intensity = 1;
                }
                geometry.colors.push(transferFunction(intensity, palette));
                averagePoint.add(p);
                count++;
              }
            }
          }
        }
      }

      console.log('max: ' + maxValue + ' min: ' + minValue);
      console.log('loaded: ' + count + ' points');

      this.cloudMaterial = buildSquareCloudMaterial(Math.exp(this.datguiSettings.size) - 1);

      return {
        object: new THREE.Points(geometry, this.cloudMaterial),
        center: averagePoint.divideScalar(count)
      };
    };
  }

  function buildPointCloud(inputData) {
    var data = new Float32Array(inputData);
    var rowLength = 2 * 3; // point and color (3 components each)
    var count = data.length / rowLength;
    var averagePoint = new THREE.Vector3(0, 0, 0);

    var geometry = new THREE.BufferGeometry();
    var positions = new Float32Array(count * 3);
    var colors = new Float32Array(count * 3);
    var filter = new Int32Array(3);
    for (var i = 0, i3 = 0; i < count; i ++, i3 += 3){
      var offset = i * rowLength;
      var x = data[offset];
      var y = data[offset + 1];
      var z = data[offset + 2];

      positions[i3 + 0] = x;
      positions[i3 + 1] = y;
      positions[i3 + 2] = z;

      offset = i * rowLength + 3;
      colors[i3 + 0] = data[offset];
      colors[i3 + 1] = data[offset + 1];
      colors[i3 + 2] = data[offset + 2];
      var p = new THREE.Vector3(x, y, z);
      averagePoint.add(p);
    }
    console.log('loaded: ' + count + ' points');
    geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.addAttribute('customColor', new THREE.BufferAttribute(colors, 3));
    geometry.addAttribute('filter', new THREE.BufferAttribute(filter, 3));
    geometry.computeBoundingBox();

    this.cloudMaterial = buildShaderCloudMaterial(Math.exp(this.datguiSettings.size) - 1);
    this.geometry = geometry;
    this.addSliceSettings();

    return {
      object: new THREE.Points(geometry, this.cloudMaterial),
      center: averagePoint.divideScalar(count)
    };
  }

  function buildVectorField(inputData) {
    var data = new Float32Array(inputData);
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

  function buildSquareCloudMaterial(size) {
    return new THREE.PointCloudMaterial({
      size: size,
      vertexColors: THREE.VertexColors,
      transparent: true,
      opacity: 1
    });
  }

  function buildShaderCloudMaterial(size){
    var c = $('<canvas width="256" height="256" />').get(0);
    var ctx = c.getContext('2d');
    ctx.beginPath();
    ctx.arc(c.width >> 1, c.height >> 1, c.width >> 1, 2 * Math.PI, false);
    ctx.fillStyle = '#ffffff';
    ctx.fill();

    var tex = new THREE.Texture(c);
    tex.needsUpdate = true;

    return new THREE.ShaderMaterial({
      uniforms: {
        color: {value: new THREE.Color(0xffffff)},
        texture: {value: tex},
        size: {value: size},
        threshold: {value: [0, 0, 0]}
      },
      vertexShader: document.getElementById('vertexshader').textContent,
      fragmentShader: document.getElementById('fragmentshader').textContent,
      transparent: true,
      blending: THREE.NormalBlending,
      depthTest: true
    });
  }

}());
