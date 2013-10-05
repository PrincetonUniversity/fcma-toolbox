/**
 * gui2fcma.js
 *
 * FCMA File Generator
 * Uses dat.gui.js as the gui for creating an .fcma file interactively
 *   bdsinger@princeton.edu
 */
 
var base = '/fastscratch/me/project/';
var analysisList = [ '1 - Voxel selection', '2 - Test prediction accuracy', '3 - Visualize correlations' ];
var blocksList = [ 'Blocks directory', 'Blocks file' ];
var classifierList = ['SVM performance', 'smart distance ratio', 'correlation sum'];
var controlWidth = 500;
var gui = new dat.GUI({ autoPlace: false, width: controlWidth});
var kClassifierSVM = 0, kClassifierSDR = 2, kClassifierSUM = 3;
var kBlocksDir = 0, kBlocksFile = 1;
var kFirstMask = 0, kSecondMask = 1, kNoMasks = 2, kBothMasks = 3;
var kSelectionIndex = 0, kTestingIndex = 1, kVisualizingIndex = 2;
var kSelectionSVM_FCMA = 0, kSelectionSDR_FCMA = 1, kSelectionSearch_MVPA = 2, kSelectionCORRSUM_FCMA = 3, kTestingTopVoxels_FCMA = 1, kTestingTopVoxels_MVPA=2, kTestingTask_FCMA = 4, kXValidateTask_FCMA = 5, kTestingTask_MVPA = 6, kXValidateTask_MVPA = 7, kVisualizationTask_FCMA = 8;
var kNA = -1;
var kTopVoxelsPrefix = base + "topvoxels";
var kTopVoxelsList = kTopVoxelsPrefix + "_list.txt";
var kTopVoxelOutputMask = kTopVoxelsPrefix + "_seq.nii.gz";
var kROIMask = base + "roi.nii.gz";
var kWholeBrainMask = base + "wholebrain.nii.gz";

var createFCMA = function() {
	var blockIsDir = (gui_params['blocksInput'] == 'Blocks directory');
	var is_mvpa_control = gui_params['mvpa_control'];
	var analysisChoice = analysisList.indexOf(gui_params["analysisStage"]);
	var classifierChoice = classifierList.indexOf(gui_params["classifier"]);
	var all_in = (params['num_items_held_for_test'] == 0);
	var num_processors = 8;
	var test_add_1 = 0;
	if (is_mvpa_control) {
		test_add_1 = 1;
	}
	if (analysisChoice == kSelectionIndex) {
		params['is_test_mode'] = 0;
		if (is_mvpa_control) {
			params['task_type'] = kSelectionSearch_MVPA;
		} else if (classifierChoice > 1) {
			params['task_type'] = kSelectionCORRSUM_FCMA;
		} else {
			params['task_type'] = kSelectionSVM_FCMA + classifierChoice;
		}
		params['first_maskfile'] = kROIMask;
		params['second_maskfile'] = kWholeBrainMask;
		params['outputfile'] = kTopVoxelsPrefix;
	} else if (analysisChoice == kTestingIndex) {
		params['is_test_mode'] = 1;
		num_processors = 1;
		if (gui_params['vary_topvoxels']) {
			params['task_type'] = kTestingTopVoxels_FCMA + test_add_1;
			params['first_maskfile'] = kWholeBrainMask;
			params['second_maskfile'] = "";
			params['outputfile'] = kTopVoxelsList;
		} else {
			if (gui_params['cross_validate']) {
				params['task_type'] = kXValidateTask_FCMA + test_add_1;
			} else {
				params['task_type'] = kTestingTask_FCMA + test_add_1;
			}
			params['first_maskfile'] = kTopVoxelOutputMask;
			params['second_maskfile'] = kWholeBrainMask;
		}
	} else if (analysisChoice == kVisualizingIndex) {
		params['first_maskfile'] = kTopVoxelOutputMask;
		params['task_type'] = kVisualizationTask_FCMA;
	}

	var res = '';
	for (k in params) {
		if ( (k == 'first_left_out_block_id') && (all_in) ) {
			res = res + k + ':-1\n';
			continue;
		}
		if (params.hasOwnProperty(k)) {
			res = res + k + ':' + params[k] + '\n';
        	}
	}
	
	res = res + '\n';
	res = res + '# check w/sysadmin before changing anything below!\n';
	res = res + '# this is the only supported fMRI format\n';
	res = res + 'matrix_format:.nii.gz\n';
	res = res + '# these are machine-specific\n';
	res = res + 'omp_num_threads:8\n';
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
  	task_type: kSelectionSVM_FCMA,
	num_folds_in_feature_selection:8,
  	first_left_out_block_id:50,
  	num_items_held_for_test:0,
  	is_test_mode:0,
	visualize_blockid:1,
  	datadir: base + 'data/',
	outputfile: kTopVoxelsPrefix,
	first_maskfile: kROIMask,
	second_maskfile: kWholeBrainMask,
  	blockdir: base + 'blockfiles/',
  	blockfile: base + 'selection.txt',
	visualize_reference: kROIMask
};

var gui_params = {
  	analysisStage: analysisList[kSelectionIndex],
  	blocksInput: blocksList[kBlocksFile],
  	classifier: classifierList[kClassifierSVM],
  	mvpa_control: false,
	vary_topvoxels: false,
	cross_validate: false
};

var changeNotifier = function (element, index, array) {
    element.onFinishChange(function(value) {
  		// Fires when a controller loses focus.
  		createFCMA();
		this.updateDisplay();
		console.log(this.property);
	});
}

var topVoxelTestChanged = function(newValue) {
	gui_params['analysisStage'] = analysisList[kTestingIndex];
	gui_params['cross_validate'] = false;
}

var crossValidateTestChanged = function(newValue) {
	gui_params['analysisStage'] = analysisList[kTestingIndex];
	gui_params['vary_topvoxels'] = false;
}

var analysisStageChangesTestTypes = function(newValue) {
	analysisChoice = analysisList.indexOf(newValue);
	if (analysisChoice !== kTestingIndex) {
		gui_params['vary_topvoxels'] = false;
		gui_params['cross_validate'] = false;
	}
}

var guiLoader = function() {
  var ctrlArray = new Array;
  ctrlArray.push( gui.add(gui_params, 'analysisStage', analysisList).name('Analysis stage').onChange(analysisStageChangesTestTypes).listen() );
  ctrlArray.push( gui.add(gui_params, 'vary_topvoxels').name('Test increasing top voxels').onChange(topVoxelTestChanged).listen());
  ctrlArray.push( gui.add(gui_params, 'cross_validate').name('Test via cross validation').onChange(crossValidateTestChanged).listen() );

  ctrlArray.push( gui.add(params, 'num_folds_in_feature_selection').min(0).max(20).step(1).name('Number of folds') );

  var blf = gui.addFolder('Holds for selection or cross validation (0 to disable)');
  ctrlArray.push( blf.add(params, 'first_left_out_block_id').min(0).max(200).step(1).name('First block held from selection (0..subjects x blocks)') );
  ctrlArray.push( blf.add(params, 'num_items_held_for_test').min(0).max(100).step(1).name('Total (select) or per fold (test)') )
  
  var mf = gui.addFolder('Cluster Files & Directories');  
  ctrlArray.push( mf.add(params, 'datadir').name('Data directory') );
  ctrlArray.push( mf.add(params, 'outputfile').name('Out prefix (select), input topvoxels (test)').listen() );

  var blockF = mf.addFolder('Blocks regressors');
  ctrlArray.push( blockF.add(gui_params,'blocksInput', blocksList).name('Blocks input') );
  ctrlArray.push( blockF.add(params,'blockdir').name('Blocks directory') );
  ctrlArray.push( blockF.add(params,'blockfile').name('Blocks file') );

 // .listen() appended to items that change based on other items
 // such as mask name in selection vs test mode
  var maskF = mf.addFolder('Voxel Mask(s)');
  ctrlArray.push( maskF.add(params,'first_maskfile').name('Mask 1').listen() );
  ctrlArray.push( maskF.add(params,'second_maskfile').name('Mask 2').listen() );

  ctrlArray.push( gui.add(gui_params, 'classifier', classifierList ).name('Classifier') );
  ctrlArray.push( gui.add(gui_params, 'mvpa_control').name('MVPA control') );
  var visF = mf.addFolder('Visualization');
  ctrlArray.push( visF.add(params, 'visualize_blockid').name('BlockID of correlations to save') );
  ctrlArray.push( visF.add(params, 'visualize_reference').name('File for reference (only for header info)') );

  mf.open()
  //maskF.open();
  blockF.open();
  //blf.open();
  
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
