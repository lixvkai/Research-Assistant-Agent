// Event delegation for the history panel:
// - clicking a `.history-item` loads the corresponding session
// - clicking the inline `×` button deletes it
// We push the session id into a hidden Gradio textbox and rely on its
// `.change` event on the Python side.

function gradioSetValue(selector, value) {
    try {
        var el = document.querySelector(selector + ' textarea')
              || document.querySelector(selector + ' input');
        if (!el) return;
        var proto = el instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
        var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
        // Append a timestamp so an identical click still triggers `change`.
        if (descriptor && descriptor.set) {
            descriptor.set.call(el, value + '|' + Date.now());
        } else {
            el.value = value + '|' + Date.now();
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    } catch (e) {
        console.warn('gradioSetValue error:', e);
    }
}

document.addEventListener('click', function (e) {
    var del = e.target.closest('.history-del');
    if (del) {
        e.stopPropagation();
        gradioSetValue('#session-deleter', del.getAttribute('data-del-sid'));
        return;
    }
    var item = e.target.closest('.history-item');
    if (item) {
        gradioSetValue('#session-loader', item.getAttribute('data-sid'));
    }
});
