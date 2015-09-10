/*
 parserWorker.js loads a neurolucida file and parses it.
 the worker espects a message data containing:
 morphology: the morphology name, which is also the name of the morphology file w/o the extension
 offset: an integer that the worker will just send back.
 TODO: This offset could be made more agnostic.
*/
'use strict';
importScripts('utils.js');
importScripts('asc.js');
self.addEventListener('message', function(e) {
  var data = e.data.morphology;
  viewerUtils.getFile('ascii/' + data + '.asc','text').then(function(s) {
    var result = {
    'asc': parser.parse(s),
    'offset': e.data.offset
  };
  self.postMessage(result);});
}, false);
