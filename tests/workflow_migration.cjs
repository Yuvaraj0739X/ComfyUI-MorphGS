const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
let extension;
const source = fs.readFileSync(path.join(__dirname, '../web/morphgs_inputs.js'), 'utf8')
    .replace(/^import .*;\r?\n/gm, '');
vm.runInNewContext(source, {app: {registerExtension(value) { extension = value; }}});

async function check(name, oldValues, expected) {
    class Node {}
    await extension.beforeRegisterNodeDef(Node, {name});
    const config = {widgets_values: oldValues};
    new Node().onConfigure(config);
    assert.equal(JSON.stringify(config.widgets_values), JSON.stringify(expected));
}
(async () => {
    for (const values of [
        ['hero.glb', 'hero', 1.6, true, 'auto_from_file_units'],
        ['hero.glb', 1.6, false, 'auto_from_file_units'],
        ['hero.glb', false], ['hero.glb'],
    ]) await check('MorphGSPreprocessCharacter', values, ['hero.glb']);
    const video = ['clip.mp4', false, 'sv4d2.safetensors', true, 24];
    await check('MorphGSPreprocessVideo', [...video, true], video);
    await check('MorphGSPreprocessVideo', [video[0], 'scene', ...video.slice(1), false], video);
    await check('MorphGSPreprocessVideo', video, video);
    console.log('Workflow migration checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
