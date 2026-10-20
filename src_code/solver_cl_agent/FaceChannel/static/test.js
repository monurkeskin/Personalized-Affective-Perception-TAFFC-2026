function load() {
    sessionStorage.clear();

    changeStep("");
}

function changeStep(step) {
    $(".btn").attr("disabled", "disabled");
    $("#cameraID").attr("disabled", "disabled");
    $("#experimentID").attr("disabled", "disabled");

    sessionStorage.setItem("step", step);

    if (step === "") {
        $("#joinBtn").removeAttr("disabled");
        $("#cameraID").removeAttr("disabled");
        $("#experimentID").removeAttr("disabled");
    } else if (step === "session1") {
        $("#startBtn").removeAttr("disabled");
        $("#nextBtn").removeAttr("disabled");
    } else if (step === "recording1" || step === "recording2") {
        $("#stopBtn").removeAttr("disabled");
    } else if (step === "session2") {
        $("#startBtn").removeAttr("disabled");
        $("#exitBtn").removeAttr("disabled");
    }
}

function updateStatus(status) {
    sessionStorage.setItem("status", status);

    $("#status").html("" + status);
    if (status === "")
        $("#status").html("Waiting...");

    if (status === false) {
        $("#status").removeClass("text-success");
        $("#status").removeClass("text-warning");
        $("#status").addClass("text-danger");

        changeStep("");
    } else if (status === "") {
        $("#status").removeClass("text-success");
        $("#status").removeClass("text-danger");
        $("#status").addClass("text-warning");

        $(".btn").attr("disabled", "disabled");
        $("#cameraID").attr("disabled", "disabled");
        $("#experimentID").attr("disabled", "disabled");
    } else {
        $("#status").removeClass("text-danger");
        $("#status").removeClass("text-warning");
        $("#status").addClass("text-success");

        changeStep(sessionStorage.getItem("step"));
    }

}

function join() {
    const experimentID = $("#experimentID").val();
    const cameraID = $("#cameraID").val();

    const request = {"id": experimentID, "camera_id": cameraID};

    updateStatus("");

    $.ajax({
        type: "POST",
        url: "join",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            if (data["status"])
                sessionStorage.setItem("step", "session1");

            updateStatus(data["status"]);
        }, fail: function () {
            updateStatus(false);
        }, error: function () {
            updateStatus(false);
        }
    });
}

function start() {
    const experimentID = $("#experimentID").val();

    const request = {"id": experimentID};

    updateStatus("");

    $.ajax({
        type: "POST",
        url: "start",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            if (data["status"])
                sessionStorage.setItem("step",
                    sessionStorage.getItem("step") === "session1" ? "recording1" : "recording2");

            updateStatus(data["status"]);
        }, fail: function () {
            updateStatus(false);
        }, error: function () {
            updateStatus(false);
        }
    });
}

function stop() {
    const experimentID = $("#experimentID").val();

    const request = {"id": experimentID};

    updateStatus("");

    $.ajax({
        type: "POST",
        url: "stop",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            if (data["status"])
                sessionStorage.setItem("step",
                    sessionStorage.getItem("step") === "recording1" ? "session1" : "session2");

            $("#arousalOut").html("" + data["predictions"]["Arousal"]);
            $("#valanceOut").html("" + data["predictions"]["Valance"]);

            updateStatus(data["status"]);
        }, fail: function () {
            updateStatus(false);
        }, error: function () {
            updateStatus(false);
        }
    });
}

function next() {
    const experimentID = $("#experimentID").val();

    const request = {"id": experimentID};

    updateStatus("");

    $.ajax({
        type: "POST",
        url: "next",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            if (data["status"])
                sessionStorage.setItem("step", "session2");

            updateStatus(data["status"]);
        }, fail: function () {
            updateStatus(false);
        }, error: function () {
            updateStatus(false);
        }
    });
}

function exit() {
    const experimentID = $("#experimentID").val();

    const request = {"id": experimentID};

    updateStatus("");

    $.ajax({
        type: "POST",
        url: "exit",
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (data) {
            if (data["status"])
                sessionStorage.setItem("step", "");

            updateStatus(data["status"]);
        }, fail: function () {
            updateStatus(false);
        }, error: function () {
            updateStatus(false);
        }
    });
}