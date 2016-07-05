'use strict';
var viewerUtils = viewerUtils ? viewerUtils : {};

(function() {
  viewerUtils.getFile = function(url, responseType) {
    console.log('loading: ' + url);

    // Return a new promise.
    return new Promise(function(resolve, reject) {
      // Do the usual XHR stuff
      var req = new XMLHttpRequest();
      req.open('GET', url);
      req.responseType = responseType;

      req.onload = function() {
        // This is called even on 404 etc
        // so check the status
        if (req.status == 200) {
          // Resolve the promise with the response text
          resolve(req.response);
        } else {
          // Otherwise reject with the status text
          // which will hopefully be a meaningful error
          reject(Error(req.statusText));
        }
      };

      // Handle network errors
      req.onerror = function() {
        reject(Error('Network Error'));
      };

      // Make the request
      req.send();
    });
  };
  // from http://stackoverflow.com/questions/901115/how-can-i-get-query-string-values-in-javascript
  viewerUtils.getParameterByName = function(name) {
    name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
    var regex = new RegExp('[\\?&]' + name + '=([^&#]*)'),
    results = regex.exec(location.search);
    var result = results === null ? '' : decodeURIComponent(results[1].replace(/\+/g, ' '));
    return result;
  };

}());