A simple Three.js based viewer for BrainBuilder related data.

It can load:
- MetaIO: Point it to the MHD file and it will pick the RAW automatically
- Point clouds

To run the viewer:
  python -m SimpleHTTPServer 8080
  open http://localhost:8080/#data/density.mhd

Coding Standard:
  https://google.github.io/styleguide/javascriptguide.xml
