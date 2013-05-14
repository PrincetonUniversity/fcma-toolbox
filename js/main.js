/*
 * jQuery File Upload Plugin JS Example 7.0
 * https://github.com/blueimp/jQuery-File-Upload
 *
 * Copyright 2010, Sebastian Tschan
 * https://blueimp.net
 *
 * Licensed under the MIT license:
 * http://www.opensource.org/licenses/MIT
 */

/*jslint nomen: true, unparam: true, regexp: true */
/*global $, window, document */

/* THIS IS CLIENT-SIDE;  console.log goes to browser's console */

$(function () {
    'use strict';

    console.log('function');

    // Initialize the jQuery File Upload widget:
    $('#fileupload').fileupload({
        // Uncomment the following to send cross-domain cookies:
        xhrFields: {withCredentials: true},
        url: '/fcmauploader/',
 		maxFileSize: 100000000, //  SET MAX FILESIZE HERE
		acceptFileTypes: /(\.|\/)(nii|nii\.gz|txt)$/i  // SET FORMATS SUPPORTED BY EXTENSION HERE
    });

    // Enable iframe cross-domain access via redirect option:
    $('#fileupload').fileupload(
        'option',
        'redirect',
        window.location.href.replace(
            /\/[^\/]*$/,
            '/cors/result.html?%s'
        )
    );

    // Load existing files:
    $.ajax({
        // Uncomment the following to send cross-domain cookies:
        xhrFields: {withCredentials: true},
        url: $('#fileupload').fileupload('option', 'url'),
        dataType: 'json',
        context: $('#fileupload')[0]
    }).done(function (result) {
        $(this).fileupload('option', 'done')
            .call(this, null, {result: result});
    });

	// callbacks available on client side
	
/*	$('#fileupload')
    .bind('fileuploadadd', function (e, data) { console.log('add'); })
    .bind('fileuploadsubmit', function (e, data) { console.log('submit'); })
    .bind('fileuploadsend', function (e, data) { console.log('send'); })
    .bind('fileuploaddone', function (e, data) { console.log('done'); })
    .bind('fileuploadfail', function (e, data) { console.log('fail'); })
    .bind('fileuploadalways', function (e, data) { console.log('always'); })
    .bind('fileuploadprogress', function (e, data) { console.log('progress'); })
    .bind('fileuploadprogressall', function (e, data) { console.log('progressall'); })
    .bind('fileuploadstart', function (e) { console.log('start'); })
    .bind('fileuploadstop', function (e) { console.log('stop'); })
    .bind('fileuploadchange', function (e, data) { console.log('change'); })
    .bind('fileuploadpaste', function (e, data) { console.log('paste'); })
    .bind('fileuploaddrop', function (e, data) { console.log('drop'); })
    .bind('fileuploaddragover', function (e) { console.log('dragover'); })
    .bind('fileuploadchunksend', function (e, data) { console.log('chunksend'); })
    .bind('fileuploadchunkdone', function (e, data) { console.log('chunkdone'); })
    .bind('fileuploadchunkfail', function (e, data) { console.log('chunkfail'); })
    .bind('fileuploadchunkalways', function (e, data) { console.log('chunkalways'); })
    .bind('fileuploaddestroy', function (e, data) { console.log('destroy'); })
    .bind('fileuploaddestroyed', function (e, data) {
		console.log('destroyed');
    }).bind('fileuploadadded', function (e, data) {
		console.log('added');
	}).bind('fileuploadsent', function (e, data) { console.log('sent'); })
    .bind('fileuploadcompleted', function (e, data) {
    }).bind('fileuploadfailed', function (e, data) {
		console.log('failed'); 
	}).bind('fileuploadfinished', function (e, data) { console.log('finished'); })
    .bind('fileuploadstarted', function (e) { console.log('started'); })
    .bind('fileuploadstopped', function (e) { 
		console.log('stopped');
	}).bind('fileuploadpreviewdone', function (e, data) { console.log('previewdone'); });
*/
});


