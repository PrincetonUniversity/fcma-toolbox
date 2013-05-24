/**
 * gui2fcma.js
 *
 * FCMA File Generator
 * Uses dat.gui.js as the gui for creating an .fcma file interactively
 */
 
var base = '/gigatmp/scratch/me/project/';
var analysisList = [ '1 - Voxel selection', '2 - Test prediction accuracy' ];
var blocksList = [ 'Blocks directory', 'Blocks file' ];
var maskList = [ 'ROI Mask 1', 'ROI Mask 2', 'No masks', 'Both masks' ];
var classifierList = ['SVM performance', 'smart distance ratio', 'correlation sum'];
var controlWidth = 500;
var gui = new dat.GUI({ autoPlace: false, width: controlWidth});
var kFirstMask = 0, kSecondMask = 1, kNoMasks = 2, kBothMasks = 3;
var kSelectionIndex = 0, kTestingIndex = 1;
var kSelectionSVM_FCMA = 0, kSelectionSDR_FCMA = 1, kSelectionSearch_MVPA = 2, kSelectionCORRSUM_FCMA = 3, kTestingTask_FCMA = 4, kXValidateTask_FCMA = 5, kTestingTask_MVPA = 6, kXValidateTask_MVPA = 7;
var kNA = -1;

var createFCMA = function() {
	var blockIsDir = (params["blocksInput"] == 'Blocks directory');
	var maskChoice = maskList.indexOf(params["maskInput"]);
	var is_mvpa_control = params['mvpa_control'];
	var analysisChoice = analysisList.indexOf(params["analysisStage"]);
	var classifierChoice = classifierList.indexOf(params["classifier"]);
	var all_in = (params['num_items_held_for_test'] == 0);
	var num_processors = 8;
	if (analysisChoice == kSelectionIndex) {
		params['is_test_mode'] = 0;
		if (is_mvpa_control) {
			params['task_type'] = kSelectionSearch_MVPA;
		} else if (classifierChoice > 1) {
			params['task_type'] = kSelectionCORRSUM_FCMA;
		} else {
			params['task_type'] = kSelectionSVM_FCMA + classifierChoice;
		}
	} else {
		params['is_test_mode'] = 1;
		num_processors = 1;
		if (is_mvpa_control) {
            params['task_type'] = kTestingTask_MVPA;
		} else {
			params['task_type'] = kTestingTask_FCMA;
		}
		if (all_in == false) {
			params['task_type'] += 1;
		}
	}

	var res = '';
	for (var k in params) {
		if ( (k == 'analysisStage') || (k == 'blocksInput') || (k == 'maskInput') || (k == 'classifier') || (k == 'mvpa_control') ) {
			continue;
		}
		if ( (k == 'blockdir') && (blockIsDir !== true) ) {
			continue;			
		}
		if ( (k == 'blockfile' ) && (blockIsDir == true) ) {
			continue;
		}
		if ( (k == 'first_maskfile') && ((maskChoice == kSecondMask) || (maskChoice == kNoMasks)) ) {
			continue;
		}
		if ( (k == 'second_maskfile' ) && ((maskChoice == kFirstMask) || (maskChoice == kNoMasks)) ) {
			continue;
		}
		if ( (k == 'first_left_out_block_id') && (all_in) ) {
			res = res + k + ':-1\n';
			continue;
		}
		
		if (typeof params[k] !== 'function') {
         	res = res + k + ':' + params[k] + '\n';
        }
	}
	
	res = res + '\n';
	res = res + '# check w/sysadmin before changing anything below!\n';
	res = res + '# this is the only supported fMRI format\n';
	res = res + 'matrix_format:.nii.gz\n';
	res = res + '# these are rondo-specific\n';
	res = res + 'omp_num_threads:7\n';
	res = res + 'num_processors:'+num_processors+'\n';
	
	var ta = document.getElementById("fcma-out");
	ta.innerHTML = res;
};

function IE(v) {
  var r = RegExp('msie' + (!isNaN(v) ? ('\\s' + v) : ''), 'i');
  return r.test(navigator.userAgent);
}

function SaveContents(element) {
    if (typeof element == "string")
        element = document.getElementById(element);
    if (element) {
        if (document.execCommand) {
            var oWin = window.open("about:blank", "_blank");
            oWin.document.write(element.value);
            oWin.document.close();
            var success = oWin.document.execCommand('SaveAs', true, ".txt");
            oWin.close();
            // if (!success)
                // alert("Sorry, your browser does not support this feature");
        }
    }
}

var downloadFCMA = function() {
	if (IE()) {
		SaveContents("fcma-out");
	} else {
		var ta = document.getElementById("fcma-out");
		window.location = "data:application/octet-stream,"+encodeURIComponent(ta.innerHTML);
	}
};

var params = {
  	analysisStage: analysisList[0],
  	maskInput: maskList[0],
  	blocksInput: blocksList[0],
  	task_type: kSelectionSVM_FCMA,
  	classifier: classifierList[0],
  	first_left_out_block_id:5,
  	num_items_held_for_test:0,
  	is_test_mode: 0,
  	mvpa_control: false,
  	datadir: base + 'data/',
	outputfile: base + 'top_correlation.txt',
	first_maskfile: base + 'masks/mask1.nii.gz',
	second_maskfile: base + 'masks/mask2.nii.gz',
  	blockdir: base + 'blockfiles/',
  	blockfile: base + 'blockfile.txt'
};
	
var changeNotifier = function (element, index, array) {
    element.onFinishChange(function(value) {
  		// Fires when a controller loses focus.
  		createFCMA();
		this.updateDisplay();
		console.log(this.property);
	});
}

var guiLoader = function() {
  var ctrlArray = new Array;
  ctrlArray.push( gui.add(params, 'analysisStage', analysisList).name('Analysis stage') );

  var blf = gui.addFolder('Blocks to leave out for cross-validation (accuracy prediction stage only)');
  ctrlArray.push( blf.add(params, 'first_left_out_block_id').min(0).max(20).step(1).name('Beginning block ID') );
  ctrlArray.push( blf.add(params, 'num_items_held_for_test').min(0).max(20).step(1).name('Number of blocks (0 to disable)') );
  
  var mf = gui.addFolder('Cluster Files & Directories');  
  ctrlArray.push( mf.add(params, 'datadir').name('Data directory') );
  ctrlArray.push( mf.add(params, 'outputfile').name('Output file') );

  var blockF = mf.addFolder('Time-series Blocks');
  ctrlArray.push( blockF.add(params,'blocksInput', blocksList).name('Blocks input') );
  ctrlArray.push( blockF.add(params,'blockdir').name('Blocks directory') );
  ctrlArray.push( blockF.add(params,'blockfile').name('Blocks file') );

  var maskF = mf.addFolder('Voxel Mask(s)');
  ctrlArray.push( maskF.add(params,'maskInput', maskList).name('Mask input') );
  ctrlArray.push( maskF.add(params,'first_maskfile').name('ROI Mask 1') );
  ctrlArray.push( maskF.add(params,'second_maskfile').name('ROI Mask 2') );

  ctrlArray.push( gui.add(params, 'classifier', classifierList ).name('Classifier') );
  ctrlArray.push( gui.add(params, 'mvpa_control').name('MVPA control condition') );
  
  mf.open();
  blockF.open();
  maskF.open();
  blf.open();
  
  ctrlArray.forEach(changeNotifier);
  
  var s = document.getElementById("fcma-gui");
  s.appendChild(gui.domElement);
  
  createFCMA();

 // gui.remember(params);
};

//http://stackoverflow.com/questions/9434/how-do-i-add-an-additional-window-onload-event-in-javascript>

if (window.addEventListener) // W3C standard
{
  window.addEventListener('load', guiLoader, false); // NB **not** 'onload'
} 
else if (window.attachEvent) // Microsoft
{
  window.attachEvent('onload', guiLoader);
}