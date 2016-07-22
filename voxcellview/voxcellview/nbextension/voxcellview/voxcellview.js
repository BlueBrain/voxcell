require.config({
  map: {
    '*': {
      'brainbuilderviewer': 'nbextensions/voxcellview/voxcellview/main',
      'bigscreen': 'nbextensions/voxcellview/extern/bigscreen.min'
    }
  },
  // shim is for lib that do not support AMD. let's manage the dependencies for them.
  shim: {
    // all trackballcontrols to listen to same event when multiple instances on the same page
    'nbextensions/voxcellview/extern/TrackballControls':
    {deps: ['nbextensions/voxcellview/extern/three.min']},
    'nbextensions/voxcellview/extern/three.min': {exports: 'THREE'},
    'nbextensions/voxcellview/voxcellview/main':
      {exports: 'brainBuilderViewer',
       deps: ['nbextensions/voxcellview/extern/three.min',
              'nbextensions/voxcellview/extern/Detector',
              'nbextensions/voxcellview/extern/TrackballControls',
              'nbextensions/voxcellview/extern/dat.gui.min',
              'nbextensions/voxcellview/voxcellview/utils',
              'nbextensions/voxcellview/extern/bigscreen.min']},
    'nbextensions/voxcellview/extern/bigscreen.min': {exports: 'BigScreen'}
  }
});

define(['nbextensions/widgets/widgets/js/widget',
        'nbextensions/widgets/widgets/js/manager',
        'base/js/utils',
        // TODO see if we can move to lodash
        'underscore',
        'brainbuilderviewer'
       ],
       function(widget, manager, utils, _, brainBuilderViewer) {

         function loadVoxcellAsBytes(name, b64, shape, dtype, bb){
           // TODO get rid of encoding.
           var bString = atob(b64);
           var bytes = new Uint8Array(bString.length);
           for (var i = 0; i < bString.length; i++){
             bytes[i] = bString.charCodeAt(i);
           }
           var reader = new FileReader();
           var promise = new Promise(function(resolve, reject) {

             reader.onloadend = function() {
               resolve(reader.result);
             };
           });
           bb.loadUrl(name, promise, shape, dtype);

           reader.readAsArrayBuffer(new Blob([bytes]));
         }

         var register = {};

         var CircuitView = widget.WidgetView.extend({

           render: function() {
             // TODO: find a way to deactivate properly the save state callback.

             this.on('displayed', this.show, this);
             this.id = utils.uuid();
             // this sets the default heights for the notebook cell
             var cellContainer = this.$el.empty()[0];
             cellContainer.style.width = '100%';
             cellContainer.style.height = '300px';
             cellContainer.style.display = 'block';

             // this can go fullscreen
             var container = document.createElement('div');
             container.style.width = '100%';
             container.style.height = '100%';
             // this is required for children absolute position
             container.style.position = 'relative';

             cellContainer.appendChild(container);

             var helperContainer = document.createElement('div');
             helperContainer.style.position = 'absolute';
             helperContainer.style.bottom = '0px';
             container.appendChild(helperContainer);

             this.bb = new brainBuilderViewer.Viewer(container, helperContainer,
                                                     this.model.get('display_parameters'));
             this.bb.init();

             var that = this;
             this.model.on('change:bytes_data', function(model, value, options) {
               loadVoxcellAsBytes(model.get('name'),
                                  model.get('bytes_data'),
                                  model.get('shape'),
                                  model.get('dtype'),
                                  that.bb);
             });

             loadVoxcellAsBytes(this.model.get('name'),
                                this.model.get('bytes_data'),
                                this.model.get('shape'),
                                this.model.get('dtype'),
                                this.bb);
           },
           show: function() {
             // certain actions can be performed only if the DOM node has been added to the document
             this.bb.onShow();
           },
           initialize: function() {
             widget.WidgetView.prototype.initialize.apply(this, arguments);
             this.scalar_properties = [];
             this.bytes_properties = [];
             this.array_properties = [];
             this.new_properties();
           },
           new_properties: function() {
             // TODO: check if this is really necessary
             this.scalar_properties.push('name');
             this.scalar_properties.push('dtype');
             this.array_properties.push('shape');
             this.bytes_properties.push('bytes_data');
           }
         });

         register.CircuitView = CircuitView;
         register.CircuitModel = widget.WidgetModel.extend({}, {
           serializers: _.extend({
               }, widget.WidgetModel.serializers)
         });

         return register;

       });
