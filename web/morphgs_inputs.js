import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const INPUT_NODE_CONFIG = {
    MorphGSPreprocessCharacter: {
        widget: "character_source_path",
        route: "/morphgs/input/characters",
        button: "Upload character",
        accept: ".fbx,.glb,.gltf,model/gltf-binary,model/gltf+json,application/octet-stream",
    },
    MorphGSPreprocessVideo: {
        widget: "video_path",
        route: "/morphgs/input/videos",
        button: "Upload video",
        accept: ".mp4,.mov,.avi,.mkv,.webm,video/*",
    },
};

function migrateLegacyWidgetValues(nodeData, config) {
    const values = config?.widgets_values;
    if (!Array.isArray(values)) return;

    // Older workflows stored a manually entered name directly after the source picker.
    // ComfyUI serializes widget values by position, so remove that obsolete value before
    // the framework maps it into the current target_height/already_masked widget.
    if (nodeData.name === "MorphGSPreprocessCharacter" &&
        typeof values[1] === "string" && typeof values[2] === "number" &&
        typeof values[3] === "boolean") {
        config.widgets_values = [values[0], values[2], values[3], ...values.slice(4)];
    }
    if (nodeData.name === "MorphGSPreprocessVideo" &&
        typeof values[1] === "string" && typeof values[2] === "boolean") {
        config.widgets_values = [values[0], ...values.slice(2)];
    }
}

async function uploadToComfyInput(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("overwrite", "false");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    return await response.json();
}

async function refreshOptions(widget, route, preferredValue) {
    const response = await api.fetchApi(route, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not scan ComfyUI input: ${response.status}`);
    const values = await response.json();
    if (!Array.isArray(values)) throw new Error("Input scan returned an invalid file list");

    widget.options.values = values;
    const next = values.includes(preferredValue) ? preferredValue :
        (values.includes(widget.value) ? widget.value : (values[0] || ""));
    widget.value = next;
    widget.callback?.(next);
    return next;
}

function addInputControls(node, config) {
    const sourceWidget = node.widgets?.find((widget) => widget.name === config.widget);
    if (!sourceWidget || sourceWidget._morphgsInputControls) return;
    sourceWidget._morphgsInputControls = true;

    const fileInput = document.createElement("input");
    Object.assign(fileInput, { type: "file", accept: config.accept, style: "display: none" });
    document.body.append(fileInput);

    const redraw = () => node.setDirtyCanvas?.(true, true);
    const refresh = async () => {
        try {
            await refreshOptions(sourceWidget, config.route);
            redraw();
        } catch (error) {
            alert(error.message || String(error));
        }
    };

    fileInput.onchange = async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        if (!file) return;
        try {
            const uploaded = await uploadToComfyInput(file);
            const uploadedPath = uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
            await refreshOptions(sourceWidget, config.route, uploadedPath);
            redraw();
        } catch (error) {
            alert(error.message || String(error));
        }
    };

    const uploadButton = node.addWidget("button", config.button, null, () => {
        app.canvas.node_widget = null;
        fileInput.click();
    });
    uploadButton.options.serialize = false;
    const refreshButton = node.addWidget("button", "Refresh input list", null, refresh);
    refreshButton.options.serialize = false;

    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        fileInput.remove();
        return originalRemoved?.apply(this, arguments);
    };

    // Scan once when the node is created, then only after an upload or an explicit refresh.
    // This avoids unwanted selection changes after a queue/generation completes.
    refresh();
}

app.registerExtension({
    name: "MorphGS.InputUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = INPUT_NODE_CONFIG[nodeData.name];
        if (!config) return;

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (serializedNode) {
            migrateLegacyWidgetValues(nodeData, serializedNode);
            return originalConfigure?.apply(this, arguments);
        };

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            addInputControls(this, config);
            return result;
        };
    },
});
