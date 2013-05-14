In this dir is the fcma file generator.

index.html: the page itself

gui2fcma.js: code that uses dat.gui.js GUI to set params written to output FCMA file

download_button.css: just some button css found online that is used for the button that will down load (via URI) the fcma file to client

dat.gui.js : slightly customized version of google's simple js param setting gui (usually for use with webgl, like GLUT, but does the job here except for being a little out of place. i  like it anyway. should probably use bootstrap controls etc but bootstrap is increasingly annoying)

fcma_setup.css: stuff that prevents bootstraps global re-definitions of everything from completely messing up dat.gui. not entirely successful; witness the select selection menus being too fat in chrome browser.

also sets up the side-by-side gui&fcma file s.t. they co-exist fairly peacefully. would be nicer if the fcma file div wrapped on mobile screens but tough cookies. this is my first css + js gui in like 15 years so shut up :)

bdsinger@princeton.edu May 2013

