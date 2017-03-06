var widget = require('jupyter-js-widgets');
var brainBuilderViewer = require('./brainBuilderViewer.js').brainBuilderViewer;
var _ = require('underscore');

function loadVoxcellAsBytes(name, bytes, shape, dtype, bb){
  var reader = new FileReader();
  var promise = new Promise(function(resolve, reject) {

    reader.onloadend = function() {
      resolve(reader.result);
    };
  });
  bb.loadUrl(name, promise, shape, dtype);

  reader.readAsArrayBuffer(new Blob([bytes]));
}

var CircuitView = widget.WidgetView.extend({
  remove: function() {
    this.bb.onRemove();
    widget.WidgetView.prototype.remove.apply(this, arguments);
  },
  render: function() {
    // TODO: find a way to deactivate properly the save state callback.

    this.on('displayed', this.show, this);
    // this sets the default heights for the notebook cell
    var cellContainer = this.$el.empty()[0];
    cellContainer.style.width = 'auto';
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
    this.model.on('change:spikes', function(model, value, options) {
      if (value && value.length > 0) {
        that.bb.initSpikeControls(value);
      }
    });
    this.model.on('change:bytes_data', function(model, value, options) {
      loadVoxcellAsBytes(model.get('name'),
                         model.get('bytes_data'),
                         model.get('shape'),
                         model.get('dtype'),
                         that.bb);
    });
    if (this.model.get('bytes_data')) {
      loadVoxcellAsBytes(this.model.get('name'),
                         this.model.get('bytes_data'),
                         this.model.get('shape'),
                         this.model.get('dtype'),
                         this.bb);
    }
    this.model.on('msg:custom', function(msg) {
      this.bb.addMorph(msg);
    }, this);
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

var CircuitModel = widget.WidgetModel.extend({}, {
  serializers: _.extend({
  }, widget.WidgetModel.serializers)
});

module.exports = {
  'CircuitModel': CircuitModel,
  'CircuitView': CircuitView
};
