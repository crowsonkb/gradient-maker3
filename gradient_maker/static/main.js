var ws;
var ws_timeout;

function ws_connect() {
    clearTimeout(ws_timeout);

    ws = new WebSocket("ws://" + window.location.host + "/websocket");

    ws.onmessage = ws_incoming;
    ws.onclose = function() { setTimeout(ws_connect, 100); };
    ws.onerror = ws.onclose;
}

function ws_incoming(e) {
    var msg = JSON.parse(e.data);

    switch(msg._) {
    case "error":
        $("#error").text(msg.text);
        $("#progress").text("");
        break;
    case "progress":
        $("#error").text("");
        $("#progress").text(msg.text);
        break;
    case "result":
        $("#error").text("");
        $("#result").html(msg.html);
        $("#download-csv").html("<a href='" + msg.downloadCsv + "'>Download as CSV</a>");
        break;
    default:
        break;
    }
}

function submit() {
    var spec = $("#grad-spec")[0].value;
    var steps = parseInt($("[name=steps]")[0].value);

    ws.send(JSON.stringify({_: "gradRequest", spec: spec, steps: steps}));
}

$(document).ready(function() {
    $("#submit").on("click", submit);
    ws_connect();
});
