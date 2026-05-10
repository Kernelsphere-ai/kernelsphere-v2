import json
import time
import hashlib
import threading
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    top: float
    right: float
    bottom: float
    left: float


@dataclass
class DOMNode:
    node_id: str
    tag_name: str
    node_type: int
    attributes: dict[str, str]
    children: list["DOMNode"]
    xpath: str
    css_selector: str
    is_visible: bool
    is_interactive: bool
    text_content: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    parent_id: Optional[str] = None


@dataclass
class AccessibilityNode:
    node_id: str
    role: str
    children: list["AccessibilityNode"]
    is_leaf: bool
    is_interactable: bool
    name: Optional[str] = None
    description: Optional[str] = None
    value: Optional[str] = None
    checked: Optional[bool | str] = None
    expanded: Optional[bool] = None
    disabled: bool = False
    required: bool = False
    selected: Optional[bool] = None
    level: Optional[int] = None
    haspopup: Optional[str] = None
    invalid: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    parent_id: Optional[str] = None
    dom_node_id: Optional[str] = None
    xpath: Optional[str] = None
    css_selector: Optional[str] = None


@dataclass
class InteractiveElement:
    tag_name: str
    role: str
    actions: list[str]
    is_visible: bool
    is_enabled: bool
    is_in_viewport: bool
    bounding_box: Optional[BoundingBox]
    dom_node_id: Optional[str] = None
    a11y_node_id: Optional[str] = None
    name: Optional[str] = None
    placeholder: Optional[str] = None
    value: Optional[str] = None
    input_type: Optional[str] = None
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    nearby_text: Optional[str] = None
    form_context: Optional[str] = None


@dataclass
class PageMeta:
    charset: Optional[str]
    language: Optional[str]
    description: Optional[str]
    keywords: Optional[str]
    og_title: Optional[str]
    og_description: Optional[str]
    canonical: Optional[str]
    robots: Optional[str]
    is_spa: bool
    has_iframes: bool
    has_shadow_dom: bool
    form_count: int
    link_count: int
    image_count: int
    script_count: int


@dataclass
class Viewport:
    width: int
    height: int
    device_pixel_ratio: float
    scroll_x: float
    scroll_y: float
    document_height: int
    document_width: int


@dataclass
class PageDiff:
    added_nodes: list[str]
    removed_nodes: list[str]
    modified_nodes: list[str]
    url_changed: bool
    title_changed: bool
    significant_change: bool


@dataclass
class PageState:
    capture_id: str
    url: str
    title: str
    timestamp: float
    viewport: Viewport
    dom_tree: DOMNode
    accessibility_tree: AccessibilityNode
    interactive_elements: list[InteractiveElement]
    meta: PageMeta
    dom_index: dict[str, DOMNode] = field(default_factory=dict)
    accessibility_index: dict[str, AccessibilityNode] = field(default_factory=dict)
    dom_to_a11y: dict[str, str] = field(default_factory=dict)
    a11y_to_dom: dict[str, str] = field(default_factory=dict)
    diff: Optional[PageDiff] = None


DOM_CAPTURE_JS = """
(function(options) {
    options = options || {};
    var includeHidden = options.includeHidden || false;
    var captureBoundingBoxes = options.captureBoundingBoxes !== false;
    var maxDepth = options.maxDepth || 40;
    var maxNodes = options.maxNodes || 6000;

    var counter = 0;

    function generateId(el) {
        if (!el.__captureId) el.__captureId = 'dom-' + (++counter);
        return el.__captureId;
    }

    function getXPath(el) {
        if (el.id) return '//*[@id="' + el.id + '"]';
        var parts = [];
        var cur = el;
        while (cur && cur.nodeType === 1) {
            var idx = 1;
            var sib = cur.previousElementSibling;
            while (sib) { if (sib.tagName === cur.tagName) idx++; sib = sib.previousElementSibling; }
            var tag = cur.tagName.toLowerCase();
            parts.unshift(idx > 1 ? tag + '[' + idx + ']' : tag);
            cur = cur.parentElement;
        }
        return '/' + parts.join('/');
    }

    function getCSSSelector(el) {
        if (el.id) return '#' + CSS.escape(el.id);
        var parts = [];
        var cur = el;
        while (cur && cur !== document.body && cur.nodeType === 1) {
            var sel = cur.tagName.toLowerCase();
            if (cur.id) { parts.unshift('#' + CSS.escape(cur.id)); break; }
            if (cur.className) {
                var classes = Array.from(cur.classList).filter(function(c) { return /^[a-zA-Z]/.test(c); }).slice(0, 2).map(function(c) { return '.' + CSS.escape(c); }).join('');
                if (classes) sel += classes;
            }
            var parent = cur.parentElement;
            if (parent) {
                var siblings = Array.from(parent.children).filter(function(s) { return s.tagName === cur.tagName; });
                if (siblings.length > 1) sel += ':nth-of-type(' + (siblings.indexOf(cur) + 1) + ')';
            }
            parts.unshift(sel);
            cur = cur.parentElement;
        }
        return parts.join(' > ') || el.tagName.toLowerCase();
    }

    function getBBox(el) {
        try {
            var r = el.getBoundingClientRect();
            return { x: r.x + window.scrollX, y: r.y + window.scrollY, width: r.width, height: r.height, top: r.top + window.scrollY, right: r.right + window.scrollX, bottom: r.bottom + window.scrollY, left: r.left + window.scrollX };
        } catch(e) { return null; }
    }

    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY') return false;
        var s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
        var r = el.getBoundingClientRect();
        return r.width > 0 || r.height > 0;
    }

    function isInteractive(el) {
        var tag = el.tagName.toLowerCase();
        if (['a','button','input','select','textarea','details','summary'].includes(tag)) return true;
        if (el.hasAttribute('onclick') || el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1') return true;
        var role = el.getAttribute('role');
        if (role && ['button','link','checkbox','radio','tab','menuitem','option','switch','textbox','combobox','listbox','slider'].includes(role)) return true;
        return window.getComputedStyle(el).cursor === 'pointer';
    }

    function captureNode(node, parentId, depth) {
        if (!node) return null;
        if (counter >= maxNodes) return null;
        if (depth > maxDepth) return null;
        if (node.nodeType === 3) {
            var t = node.textContent.trim();
            if (!t) return null;
            return { nodeId: 'txt-' + (++counter), nodeType: 3, tagName: '#text', textContent: t, attributes: {}, children: [], parentId: parentId, isVisible: true, isInteractive: false, xpath: '', cssSelector: '' };
        }
        if (node.nodeType !== 1) return null;
        var el = node;
        var tag = el.tagName.toLowerCase();
        if (['script','style','noscript','head'].includes(tag)) return null;
        var visible = isVisible(el);
        if (!includeHidden && !visible && !el.getAttribute('role')) return null;
        var id = generateId(el);
        var attrs = {};
        for (var i = 0; i < el.attributes.length; i++) attrs[el.attributes[i].name] = el.attributes[i].value;
        var captured = { nodeId: id, nodeType: 1, tagName: tag, attributes: attrs, children: [], parentId: parentId, isVisible: visible, isInteractive: isInteractive(el), xpath: getXPath(el), cssSelector: getCSSSelector(el) };
        if (el.childNodes.length === 0) captured.textContent = el.textContent.trim();
        if (captureBoundingBoxes) captured.boundingBox = getBBox(el);
        for (var j = 0; j < el.childNodes.length; j++) {
            var child = captureNode(el.childNodes[j], id, depth + 1);
            if (child) captured.children.push(child);
        }
        return captured;
    }

    var root = document.body || document.documentElement;
    var tree = captureNode(root, null, 0);
    return { tree: tree, nodeCount: counter, url: window.location.href, title: document.title, viewport: { width: window.innerWidth, height: window.innerHeight, devicePixelRatio: window.devicePixelRatio, scrollX: window.scrollX, scrollY: window.scrollY, documentHeight: document.documentElement.scrollHeight, documentWidth: document.documentElement.scrollWidth } };
})
"""

A11Y_CAPTURE_JS = r"""
(function() {
    var counter = 0;

    function getImplicitRole(el) {
        var tag = el.tagName.toLowerCase();
        var type = (el.getAttribute('type') || '').toLowerCase();
        var map = { button: 'button', select: 'listbox', textarea: 'textbox', nav: 'navigation', main: 'main', header: 'banner', footer: 'contentinfo', aside: 'complementary', section: 'region', form: 'form', table: 'table', tr: 'row', th: 'columnheader', td: 'cell', dialog: 'dialog', details: 'group', summary: 'button', ul: 'list', ol: 'list', li: 'listitem' };
        if (tag === 'input') {
            var inputMap = { checkbox: 'checkbox', radio: 'radio', range: 'slider', number: 'spinbutton', search: 'searchbox', button: 'button', submit: 'button', reset: 'button' };
            return inputMap[type] || 'textbox';
        }
        if (tag === 'a') return el.hasAttribute('href') ? 'link' : 'generic';
        if (/^h[1-6]$/.test(tag)) return 'heading';
        return map[tag] || 'generic';
    }

    function getAccessibleName(el) {
        var lb = el.getAttribute('aria-labelledby');
        if (lb) { var n = lb.split(' ').map(function(id) { var r = document.getElementById(id); return r ? r.textContent.trim() : ''; }).join(' ').trim(); if (n) return n; }
        var al = el.getAttribute('aria-label');
        if (al) return al.trim();
        if (el.id) { var lbl = document.querySelector('label[for="' + el.id + '"]'); if (lbl) return lbl.textContent.trim(); }
        var wl = el.closest('label');
        if (wl) { var cl = wl.cloneNode(true); cl.querySelectorAll('input,select,textarea').forEach(function(i) { i.remove(); }); var wt = cl.textContent.trim(); if (wt) return wt; }
        var title = el.getAttribute('title');
        if (title) return title.trim();
        var alt = el.getAttribute('alt');
        if (alt) return alt.trim();
        var ph = el.getAttribute('placeholder');
        if (ph) return ph.trim();
        var txt = el.textContent.trim().replace(/\s+/g, ' ').substring(0, 100);
        return txt || null;
    }

    function getBBox(el) {
        try { var r = el.getBoundingClientRect(); return { x: r.x + window.scrollX, y: r.y + window.scrollY, width: r.width, height: r.height, top: r.top + window.scrollY, right: r.right + window.scrollX, bottom: r.bottom + window.scrollY, left: r.left + window.scrollX }; } catch(e) { return null; }
    }

    var interactableRoles = ['button','link','checkbox','radio','textbox','searchbox','spinbutton','slider','switch','tab','menuitem','option','combobox','listbox','treeitem','gridcell'];

    function captureNode(el, parentId, depth) {
        if (depth > 50 || !el || el.nodeType !== 1) return null;
        var tag = el.tagName.toLowerCase();
        if (['script','style','noscript','meta','head'].includes(tag)) return null;
        if (el.getAttribute('aria-hidden') === 'true') return null;
        var role = el.getAttribute('role') || getImplicitRole(el);
        if (role === 'presentation' || role === 'none') return null;
        var id = 'a11y-' + (++counter);
        var name = getAccessibleName(el);
        var db = el.getAttribute('aria-describedby');
        var desc = db ? db.split(' ').map(function(did) { var r = document.getElementById(did); return r ? r.textContent.trim() : ''; }).join(' ').trim() : null;
        var level = null;
        var lv = el.getAttribute('aria-level');
        if (lv) { level = parseInt(lv); } else { var m = el.tagName.match(/^H([1-6])$/i); if (m) level = parseInt(m[1]); }
        var node = {
            nodeId: id, role: role, name: name || null, description: desc || null,
            value: el.value !== undefined ? String(el.value) : null,
            checked: el.checked !== undefined ? el.checked : (el.getAttribute('aria-checked') === 'mixed' ? 'mixed' : el.getAttribute('aria-checked') === 'true' ? true : el.getAttribute('aria-checked') === 'false' ? false : null),
            expanded: el.getAttribute('aria-expanded') !== null ? el.getAttribute('aria-expanded') === 'true' : null,
            disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
            required: !!(el.required || el.getAttribute('aria-required') === 'true'),
            selected: el.getAttribute('aria-selected') !== null ? el.getAttribute('aria-selected') === 'true' : null,
            level: level, haspopup: el.getAttribute('aria-haspopup') || null,
            invalid: el.getAttribute('aria-invalid') || null,
            children: [], parentId: parentId, isLeaf: false,
            isInteractable: interactableRoles.includes(role),
            domNodeId: el.__captureId || null, boundingBox: getBBox(el)
        };
        for (var i = 0; i < el.children.length; i++) {
            var child = captureNode(el.children[i], id, depth + 1);
            if (child) node.children.push(child);
        }
        node.isLeaf = node.children.length === 0;
        return node;
    }

    var root = document.body || document.documentElement;
    var tree = captureNode(root, null, 0);
    return { tree: tree, nodeCount: counter };
})
"""

INTERACTIVE_JS = r"""
(function() {
    var selectors = ['a[href]','button:not([disabled])','input:not([type=hidden]):not([disabled])','select:not([disabled])','textarea:not([disabled])','[role="button"]:not([aria-disabled="true"])','[role="link"]','[role="checkbox"]','[role="radio"]','[role="tab"]','[role="menuitem"]','[role="option"]','[role="switch"]','[role="textbox"]','[role="combobox"]','[role="slider"]','[tabindex]:not([tabindex="-1"])','[onclick]'];
    var seen = new WeakSet();
    var found = document.querySelectorAll(selectors.join(','));
    var results = [];

    function getLabelText(el) {
        var al = el.getAttribute('aria-label'); if (al) return al.trim();
        var lb = el.getAttribute('aria-labelledby');
        if (lb) return lb.split(' ').map(function(id) { var r = document.getElementById(id); return r ? r.textContent.trim() : ''; }).join(' ').trim() || null;
        if (el.id) { var lbl = document.querySelector('label[for="' + el.id + '"]'); if (lbl) return lbl.textContent.trim(); }
        return el.textContent.trim().substring(0, 80) || el.getAttribute('placeholder') || el.getAttribute('title') || null;
    }

    function getNearbyText(el) {
        var p = el.parentElement;
        return p ? p.textContent.trim().replace(/\s+/g, ' ').substring(0, 150) : null;
    }

    function getFormContext(el) {
        var f = el.closest('form');
        return f ? (f.id || f.getAttribute('aria-label') || f.getAttribute('name') || 'form') : null;
    }

    // Selector generation: priority chain
    // Generates the most specific stable selector available, in priority order:
    // #id -> [data-testid/data-cy/data-qa] -> tag[aria-label] -> tag[name]
    // -> tag[placeholder] -> a[href] -> #ancestor child-path -> semantic classes
    // This avoids the broken tag+utility-class approach that produces identical
    // selectors for all Booking.com buttons (de576f5064, dc15842869, etc.).
    function esc(s) { return s.replace(/\\/g,'\\\\').replace(/"/g,'\\"'); }

    function buildRelSel(el, ancestor) {
        var parts = []; var cur = el; var limit = 4;
        while (cur && cur !== ancestor && limit-- > 0) {
            var t = cur.tagName.toLowerCase();
            var al = cur.getAttribute('aria-label');
            if (al && al.length <= 60) { parts.unshift(t + '[aria-label="' + esc(al) + '"]'); break; }
            var tid = cur.getAttribute('data-testid');
            if (tid) { parts.unshift('[data-testid="' + esc(tid) + '"]'); break; }
            var parent = cur.parentElement;
            if (parent && parent !== ancestor) {
                var sibs = Array.from(parent.children).filter(function(s){return s.tagName===cur.tagName;});
                parts.unshift(sibs.length > 1 ? t+':nth-of-type('+(sibs.indexOf(cur)+1)+')' : t);
            } else { parts.unshift(t); }
            cur = parent;
        }
        return parts.slice(0,3).join(' > ');
    }

    function getCSSSelector(el) {
        var tag = el.tagName.toLowerCase();
        // 1. Explicit ID - globally unique
        if (el.id) return '#' + CSS.escape(el.id);
        // 2. Common semantic test attributes (data-testid, data-cy, data-qa, etc.)
        var testAttrs = ['data-testid','data-test','data-cy','data-qa','data-automation-id'];
        for (var i=0;i<testAttrs.length;i++){var v=el.getAttribute(testAttrs[i]);if(v)return '['+testAttrs[i]+'="'+esc(v)+'"]';}
        // 3. aria-label (meaningful for any interactive element)
        var al = el.getAttribute('aria-label');
        if (al && al.length <= 80) return tag+'[aria-label="'+esc(al)+'"]';
        // 4. name attribute (form fields, inputs)
        var nm = el.getAttribute('name');
        if (nm) return tag+'[name="'+esc(nm)+'"]';
        // 5. placeholder (text inputs / comboboxes)
        var ph = el.getAttribute('placeholder');
        if (ph && ph.length <= 60) return tag+'[placeholder="'+esc(ph)+'"]';
        // 6. href for anchor links (stable enough for site-internal links)
        if (tag==='a'){var href=el.getAttribute('href');if(href&&href.length<=120&&href.indexOf('javascript:')<0)return 'a[href="'+esc(href)+'"]';}
        // 7. Nearest ID-bearing ancestor + short relative path
        var anc=el.parentElement, depth=0;
        while(anc&&anc!==document.body&&depth<4){
            if(anc.id){
                var rel=buildRelSel(el,anc);
                if(rel){
                    var full='#'+CSS.escape(anc.id)+' '+rel;
                    try{if(document.querySelectorAll(full).length===1)return full;}catch(e){}
                }
            }
            anc=anc.parentElement; depth++;
        }
        // 8. Semantic class names only (filter out hash/utility classes)
        var semCls=Array.from(el.classList).filter(function(c){
            return c.length>=2&&c.length<=25&&!/^[a-f0-9]{6,}$/.test(c)&&!/^[a-zA-Z0-9]{10,}$/.test(c);
        });
        if(semCls.length>0)return tag+'.'+semCls.slice(0,2).map(function(c){return CSS.escape(c);}).join('.');
        // Fallback: bare tag (least specific - avoided by all steps above)
        return tag;
    }

    function getXPath(el) {
        if (el.id) return '//*[@id="' + el.id.replace(/"/g,'&quot;') + '"]';
        var tid = el.getAttribute('data-testid');
        if (tid) return '//*[@data-testid="' + tid.replace(/"/g,'&quot;') + '"]';
        var al = el.getAttribute('aria-label');
        if (al) return '//' + el.tagName.toLowerCase() + '[@aria-label="' + al.replace(/"/g,'&quot;') + '"]';
        return '';
    }

    for (var i = 0; i < found.length; i++) {
        var el = found[i];
        if (seen.has(el)) continue;
        seen.add(el);
        var rect = el.getBoundingClientRect();
        var tag = el.tagName.toLowerCase();
        var type = el.getAttribute('type') || '';
        var role = el.getAttribute('role') || tag;
        var actions = [];
        if (['button','a','div','span'].includes(tag) || role === 'button' || role === 'link') actions.push('click');
        if (['input','textarea'].includes(tag) && !['checkbox','radio','submit','button','file'].includes(type)) actions.push('type');
        if (tag === 'select' || role === 'listbox' || role === 'combobox') actions.push('select');
        if (type === 'checkbox' || role === 'checkbox' || role === 'switch') { actions.push('check'); actions.push('uncheck'); }
        if (type === 'radio' || role === 'radio') actions.push('check');
        if (type === 'file') actions.push('upload');
        if (actions.length === 0) actions.push('click');
        results.push({
            domNodeId: el.__captureId || null, role: role, tagName: tag,
            inputType: type || null, name: getLabelText(el),
            placeholder: el.getAttribute('placeholder') || null,
            value: el.value !== undefined ? String(el.value) : null,
            xpath: getXPath(el),
            cssSelector: getCSSSelector(el),
            boundingBox: { x: rect.x + window.scrollX, y: rect.y + window.scrollY, width: rect.width, height: rect.height, top: rect.top + window.scrollY, right: rect.right + window.scrollX, bottom: rect.bottom + window.scrollY, left: rect.left + window.scrollX },
            isInViewport: rect.top >= 0 && rect.left >= 0 && rect.bottom <= window.innerHeight && rect.right <= window.innerWidth,
            isVisible: rect.width > 0 && rect.height > 0,
            isEnabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
            actions: actions, nearbyText: getNearbyText(el), formContext: getFormContext(el)
        });
    }
    return results;
})
"""

META_JS = """
(function() {
    function getMeta(name) { var el = document.querySelector('meta[name="' + name + '"], meta[property="' + name + '"]'); return el ? el.getAttribute('content') : null; }
    var canonical = document.querySelector('link[rel="canonical"]');
    return {
        charset: document.characterSet || null,
        language: document.documentElement.lang || null,
        description: getMeta('description'),
        keywords: getMeta('keywords'),
        ogTitle: getMeta('og:title'),
        ogDescription: getMeta('og:description'),
        canonical: canonical ? canonical.href : null,
        robots: getMeta('robots'),
        isSpa: !!(window.__REACT_DEVTOOLS_GLOBAL_HOOK__ || window.__VUE__ || window.angular || window.Ember),
        hasIframes: document.querySelectorAll('iframe').length > 0,
        hasShadowDom: false,
        formCount: document.forms.length,
        linkCount: document.links.length,
        imageCount: document.images.length,
        scriptCount: document.scripts.length
    };
})
"""


def _parse_bbox(data: Optional[dict]) -> Optional[BoundingBox]:
    if not data:
        return None
    return BoundingBox(
        x=data["x"], y=data["y"], width=data["width"], height=data["height"],
        top=data["top"], right=data["right"], bottom=data["bottom"], left=data["left"]
    )


def _parse_dom_node(data: Optional[dict]) -> DOMNode:
    if not isinstance(data, dict):
        return DOMNode(
            node_id="dom-root-fallback",
            tag_name="html",
            node_type=1,
            attributes={},
            children=[],
            xpath="/html",
            css_selector="html",
            is_visible=True,
            is_interactive=False,
            text_content=None,
            bounding_box=None,
            parent_id=None,
        )
    return DOMNode(
        node_id=str(data.get("nodeId") or "dom-node-unknown"),
        tag_name=str(data.get("tagName") or "div"),
        node_type=int(data.get("nodeType") or 1),
        attributes=data.get("attributes", {}),
        children=[_parse_dom_node(c) for c in data.get("children", []) if isinstance(c, dict)],
        xpath=str(data.get("xpath") or ""),
        css_selector=str(data.get("cssSelector") or ""),
        is_visible=data.get("isVisible", True),
        is_interactive=data.get("isInteractive", False),
        text_content=data.get("textContent"),
        bounding_box=_parse_bbox(data.get("boundingBox")),
        parent_id=data.get("parentId"),
    )


def _parse_a11y_node(data: dict) -> AccessibilityNode:
    return AccessibilityNode(
        node_id=data["nodeId"],
        role=data["role"],
        children=[_parse_a11y_node(c) for c in data.get("children", [])],
        is_leaf=data.get("isLeaf", False),
        is_interactable=data.get("isInteractable", False),
        name=data.get("name"),
        description=data.get("description"),
        value=data.get("value"),
        checked=data.get("checked"),
        expanded=data.get("expanded"),
        disabled=data.get("disabled", False),
        required=data.get("required", False),
        selected=data.get("selected"),
        level=data.get("level"),
        haspopup=data.get("haspopup"),
        invalid=data.get("invalid"),
        bounding_box=_parse_bbox(data.get("boundingBox")),
        parent_id=data.get("parentId"),
        dom_node_id=data.get("domNodeId"),
        xpath=data.get("xpath"),
        css_selector=data.get("cssSelector"),
    )


def _parse_interactive(data: dict) -> InteractiveElement:
    if not isinstance(data, dict):
        return InteractiveElement(
            tag_name="div",
            role="generic",
            actions=[],
            is_visible=False,
            is_enabled=False,
            is_in_viewport=False,
            bounding_box=None,
        )
    return InteractiveElement(
        tag_name=str(data.get("tagName") or "div"),
        role=str(data.get("role") or "generic"),
        actions=list(data.get("actions") or []),
        is_visible=data.get("isVisible", True),
        is_enabled=data.get("isEnabled", True),
        is_in_viewport=data.get("isInViewport", False),
        bounding_box=_parse_bbox(data.get("boundingBox")),
        dom_node_id=data.get("domNodeId"),
        name=data.get("name"),
        placeholder=data.get("placeholder"),
        value=data.get("value"),
        input_type=data.get("inputType"),
        xpath=data.get("xpath"),
        css_selector=data.get("cssSelector"),
        nearby_text=data.get("nearbyText"),
        form_context=data.get("formContext"),
    )


def _build_dom_index(node: DOMNode, index: dict[str, DOMNode]) -> None:
    index[node.node_id] = node
    for child in node.children:
        _build_dom_index(child, index)


def _build_a11y_index(node: AccessibilityNode, index: dict[str, AccessibilityNode]) -> None:
    index[node.node_id] = node
    for child in node.children:
        _build_a11y_index(child, index)


def _build_cross_refs(
    node: AccessibilityNode,
    dom_index: dict[str, DOMNode],
    dom_to_a11y: dict[str, str],
    a11y_to_dom: dict[str, str],
) -> None:
    if node.dom_node_id and node.dom_node_id in dom_index:
        dom_to_a11y[node.dom_node_id] = node.node_id
        a11y_to_dom[node.node_id] = node.dom_node_id
        dom_node = dom_index[node.dom_node_id]
        node.xpath = dom_node.xpath
        node.css_selector = dom_node.css_selector
    for child in node.children:
        _build_cross_refs(child, dom_index, dom_to_a11y, a11y_to_dom)


class StateDifferentiator:
    def __init__(self):
        self._prev: Optional[PageState] = None

    def compare(self, current: PageState) -> PageState:
        if not self._prev:
            self._prev = current
            return current
        prev = self._prev
        added = [nid for nid in current.dom_index if nid not in prev.dom_index]
        removed = [nid for nid in prev.dom_index if nid not in current.dom_index]
        modified = []
        for nid in current.dom_index:
            if nid in prev.dom_index:
                cn = current.dom_index[nid]
                pn = prev.dom_index[nid]
                if cn.attributes != pn.attributes or cn.text_content != pn.text_content:
                    modified.append(nid)
        current.diff = PageDiff(
            added_nodes=added,
            removed_nodes=removed,
            modified_nodes=modified,
            url_changed=prev.url != current.url,
            title_changed=prev.title != current.title,
            significant_change=len(added) + len(removed) > 10 or prev.url != current.url,
        )
        self._prev = current
        return current

    def reset(self):
        self._prev = None


class CaptureCache:
    """In-memory URL -> PageState cache with optional disk persistence.

    Disk writes are **disabled by default**.  On a benchmark run with 800+ tasks
    each PageState serialises to 2-4 MB; writing them all accumulates ~2 GB of
    files that are never read (the context is reset between tasks).  Pass
    ``enable_disk_writes=True`` only in explicit offline-caching workflows.
    """

    def __init__(self, cache_dir: str = "./cache", enable_disk_writes: bool = False):
        self._dir = Path(cache_dir)
        self._enable_disk = enable_disk_writes
        if self._enable_disk:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, PageState] = {}

    def get(self, key: str) -> Optional[PageState]:
        return self._mem.get(key)

    def set(self, key: str, state: PageState) -> None:
        self._mem[key] = state
        if not self._enable_disk:
            return
        safe_key = hashlib.md5(key.encode()).hexdigest()
        path = self._dir / f"{safe_key}.json"
        path.write_text(json.dumps(asdict(state), indent=2, default=str))

    def invalidate(self, key: str) -> None:
        self._mem.pop(key, None)
        safe_key = hashlib.md5(key.encode()).hexdigest()
        path = self._dir / f"{safe_key}.json"
        if path.exists():
            path.unlink()


class ExtractionEngine:
    def __init__(self, headless: bool = True):
        self._headless = headless
        self._pw = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._differentiator = StateDifferentiator()
        self._cache = CaptureCache()
        self._context_profile_index = 0
        # Load proxy config once at construction time
        try:
            from config import get_config
            self._proxy_cfg = get_config().proxy
        except Exception:
            self._proxy_cfg = None

    def start(self) -> "ExtractionEngine":
        _proxy = self._proxy_cfg.playwright_proxy() if self._proxy_cfg else None
        logger.info(
            "Starting browser (headless=%s, proxy=%s)",
            self._headless,
            f"{self._proxy_cfg.host}:{self._proxy_cfg.port}" if _proxy else "none",
        )
        self._pw = sync_playwright().start()
        _launch_kwargs: dict = {
            "headless": self._headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        }
        # Proxy at launch level sets a default for all contexts; context-level
        # proxy overrides per-context (used in new_context below for rotation).
        if _proxy:
            _launch_kwargs["proxy"] = _proxy
        self._browser = self._pw.chromium.launch(**_launch_kwargs)
        self._context = self.new_context()
        return self

    def new_context(self, anti_bot_mode: bool = False) -> BrowserContext:
        if self._browser is None:
            raise RuntimeError("Browser is not started.")
        profiles = [
            {
                "viewport": {"width": 1280, "height": 720},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "locale": "en-US",
                "timezone_id": "America/Los_Angeles",
                "color_scheme": "light",
            },
            {
                "viewport": {"width": 1366, "height": 768},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "color_scheme": "light",
            },
            {
                "viewport": {"width": 1440, "height": 900},
                "user_agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "locale": "en-US",
                "timezone_id": "America/Chicago",
                "color_scheme": "light",
            },
        ]
        if anti_bot_mode:
            self._context_profile_index = (self._context_profile_index + 1) % len(profiles)
        p = profiles[self._context_profile_index]
        _proxy = self._proxy_cfg.playwright_proxy() if self._proxy_cfg else None
        _ctx_kwargs: dict = {
            "viewport": p["viewport"],
            "user_agent": p["user_agent"],
            "locale": p["locale"],
            "timezone_id": p["timezone_id"],
            "color_scheme": p["color_scheme"],
            "device_scale_factor": 1.0,
        }
        if _proxy:
            _ctx_kwargs["proxy"] = _proxy
        ctx = self._browser.new_context(**_ctx_kwargs)
        try:
            ctx.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
            })
        except Exception:
            pass
        try:
            ctx.add_init_script(
                """
                (function() {
                    try { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); } catch(e) {}
                    try { Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] }); } catch(e) {}
                    try { Object.defineProperty(navigator, 'platform', { get: () => 'Win32' }); } catch(e) {}
                    try { Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 }); } catch(e) {}
                    try { Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 }); } catch(e) {}
                })();
                """
            )
        except Exception:
            pass
        # Light "human-like" jitter to reduce identical startup cadence.
        try:
            page = ctx.new_page()
            page.wait_for_timeout(150 + random.randint(0, 180))
            page.close()
        except Exception:
            pass
        return ctx

    def stop(self) -> None:
        logger.info("Stopping browser.")
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def __enter__(self) -> "ExtractionEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def capture_url(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = 30000,
        use_cache: bool = False,
        include_hidden: bool = False,
        capture_bboxes: bool = True,
    ) -> PageState:
        if use_cache:
            cached = self._cache.get(url)
            if cached:
                return cached

        if self._context is None:
            raise RuntimeError("ExtractionEngine not started; call start() first.")
        logger.debug("Capturing URL: %s", url)
        page = self._context.new_page()
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout)
            state = self._capture_page(page, include_hidden=include_hidden, capture_bboxes=capture_bboxes)
        finally:
            page.close()

        if use_cache:
            self._cache.set(url, state)

        return self._differentiator.compare(state)

    def capture_page(
        self,
        page: Page,
        include_hidden: bool = False,
        capture_bboxes: bool = True,
    ) -> PageState:
        state = self._capture_page(page, include_hidden=include_hidden, capture_bboxes=capture_bboxes)
        return self._differentiator.compare(state)

    def _capture_page(self, page: Page, include_hidden: bool, capture_bboxes: bool) -> PageState:
        capture_id = f"cap-{int(time.time() * 1000)}"

        dom_result = page.evaluate(
            f"({DOM_CAPTURE_JS})({{ includeHidden: {str(include_hidden).lower()}, captureBoundingBoxes: {str(capture_bboxes).lower()}, maxDepth: 36, maxNodes: 5000 }})"
        )
        if not isinstance(dom_result, dict):
            dom_result = {}

        try:
            a11y_result = page.evaluate(f"({A11Y_CAPTURE_JS})()")
            a11y_tree = _parse_a11y_node(a11y_result["tree"]) if a11y_result.get("tree") else None
        except Exception:
            a11y_tree = None

        interactive_raw = page.evaluate(f"({INTERACTIVE_JS})()")
        if not isinstance(interactive_raw, list):
            interactive_raw = []
        meta_raw = page.evaluate(f"({META_JS})()")
        if not isinstance(meta_raw, dict):
            meta_raw = {}

        dom_tree = _parse_dom_node(dom_result.get("tree"))
        vp_raw = dom_result.get("viewport") if isinstance(dom_result.get("viewport"), dict) else {}
        _vp_size = page.viewport_size or {"width": 1280, "height": 720}
        viewport = Viewport(
            width=int(vp_raw.get("width", _vp_size.get("width", 1280))),
            height=int(vp_raw.get("height", _vp_size.get("height", 720))),
            device_pixel_ratio=float(vp_raw.get("devicePixelRatio", 1.0)),
            scroll_x=float(vp_raw.get("scrollX", 0.0)),
            scroll_y=float(vp_raw.get("scrollY", 0.0)),
            document_height=int(vp_raw.get("documentHeight", _vp_size.get("height", 720))),
            document_width=int(vp_raw.get("documentWidth", _vp_size.get("width", 1280))),
        )
        meta = PageMeta(
            charset=meta_raw.get("charset"),
            language=meta_raw.get("language"),
            description=meta_raw.get("description"),
            keywords=meta_raw.get("keywords"),
            og_title=meta_raw.get("ogTitle"),
            og_description=meta_raw.get("ogDescription"),
            canonical=meta_raw.get("canonical"),
            robots=meta_raw.get("robots"),
            is_spa=meta_raw.get("isSpa", False),
            has_iframes=meta_raw.get("hasIframes", False),
            has_shadow_dom=meta_raw.get("hasShadowDom", False),
            form_count=meta_raw.get("formCount", 0),
            link_count=meta_raw.get("linkCount", 0),
            image_count=meta_raw.get("imageCount", 0),
            script_count=meta_raw.get("scriptCount", 0),
        )
        interactive_elements = [_parse_interactive(e) for e in interactive_raw if isinstance(e, dict)]

        dom_index: dict[str, DOMNode] = {}
        a11y_index: dict[str, AccessibilityNode] = {}
        dom_to_a11y: dict[str, str] = {}
        a11y_to_dom: dict[str, str] = {}

        _build_dom_index(dom_tree, dom_index)
        if a11y_tree:
            _build_a11y_index(a11y_tree, a11y_index)
            _build_cross_refs(a11y_tree, dom_index, dom_to_a11y, a11y_to_dom)

        try:
            _title = page.title()
        except Exception:
            _title = ""

        return PageState(
            capture_id=capture_id,
            url=page.url,
            title=_title,
            timestamp=time.time(),
            viewport=viewport,
            dom_tree=dom_tree,
            accessibility_tree=a11y_tree,
            interactive_elements=interactive_elements,
            meta=meta,
            dom_index=dom_index,
            accessibility_index=a11y_index,
            dom_to_a11y=dom_to_a11y,
            a11y_to_dom=a11y_to_dom,
        )

    def _convert_playwright_a11y(self, node: dict, parent_id: Optional[str], counter: list[int]) -> AccessibilityNode:
        counter[0] += 1
        node_id = f"a11y-{counter[0]}"
        interactable_roles = {
            "button", "link", "checkbox", "radio", "textbox", "searchbox",
            "spinbutton", "slider", "switch", "tab", "menuitem", "option",
            "combobox", "listbox",
        }
        role = node.get("role", "generic")
        children_raw = node.get("children", [])
        children = [self._convert_playwright_a11y(c, node_id, counter) for c in children_raw]
        value = node.get("value")
        return AccessibilityNode(
            node_id=node_id,
            role=role,
            name=node.get("name") or None,
            description=node.get("description") or None,
            value=str(value) if value is not None else None,
            checked=node.get("checked"),
            expanded=node.get("expanded"),
            disabled=node.get("disabled", False),
            required=node.get("required", False),
            selected=node.get("selected"),
            level=node.get("level"),
            haspopup=node.get("haspopup") or None,
            invalid=node.get("invalid") or None,
            children=children,
            parent_id=parent_id,
            is_leaf=len(children) == 0,
            is_interactable=role in interactable_roles,
        )

    def reset_diff(self) -> None:
        self._differentiator.reset()

    def print_dom_tree(self, node: DOMNode, depth: int = 0, max_depth: int = 5) -> str:
        if depth > max_depth:
            return ""
        indent = "  " * depth
        keep_attrs = {"id", "class", "type", "role", "aria-label", "href", "name", "placeholder"}
        attrs = " ".join(
            f'{k}="{v[:30]}"' for k, v in node.attributes.items() if k in keep_attrs
        )
        text = f' "{node.text_content[:40]}"' if node.text_content else ""
        flags = (" [i]" if node.is_interactive else "") + ("" if node.is_visible else " [h]")
        out = f"{indent}<{node.tag_name}{(' ' + attrs) if attrs else ''}>{text}{flags}\n"
        for child in node.children:
            out += self.print_dom_tree(child, depth + 1, max_depth)
        return out

    def print_a11y_tree(self, node: AccessibilityNode, depth: int = 0, max_depth: int = 5) -> str:
        if depth > max_depth:
            return ""
        indent = "  " * depth
        name = f' "{node.name}"' if node.name else ""
        value = f' val="{node.value}"' if node.value else ""
        states = [s for s in [
            f"checked={node.checked}" if node.checked is not None else None,
            f"expanded={node.expanded}" if node.expanded is not None else None,
            "disabled" if node.disabled else None,
            "required" if node.required else None,
        ] if s]
        state_str = f" [{', '.join(states)}]" if states else ""
        marker = " *" if node.is_interactable else ""
        out = f"{indent}[{node.role}]{name}{value}{state_str}{marker}\n"
        for child in node.children:
            out += self.print_a11y_tree(child, depth + 1, max_depth)
        return out


if __name__ == "__main__":
    TEST_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Capture engine test page">
  <title>Capture Test</title>
</head>
<body>
  <header>
    <h1>Test Page</h1>
    <nav aria-label="Main nav">
      <a href="/home">Home</a>
      <a href="/about">About</a>
    </nav>
  </header>
  <main>
    <form id="login-form" aria-label="Login">
      <label for="username">Username</label>
      <input id="username" type="text" name="username" placeholder="Enter username" required>
      <label for="password">Password</label>
      <input id="password" type="password" name="password" placeholder="Enter password" required>
      <label for="role-select">Role</label>
      <select id="role-select" name="role">
        <option value="user">User</option>
        <option value="admin">Admin</option>
      </select>
      <input type="checkbox" id="remember" name="remember" aria-label="Remember me">
      <label for="remember">Remember me</label>
      <button type="submit">Sign In</button>
      <button type="reset">Clear</button>
    </form>
    <section aria-labelledby="products-heading">
      <h2 id="products-heading">Products</h2>
      <ul>
        <li><a href="/product/1">Product A</a> - <button aria-label="Add Product A to cart">Add to Cart</button></li>
        <li><a href="/product/2">Product B</a> - <button aria-label="Add Product B to cart">Add to Cart</button></li>
      </ul>
    </section>
    <div role="alert" aria-live="polite" style="display:none">Saved successfully</div>
  </main>
</body>
</html>"""

    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write(TEST_HTML)
        tmp_path = f.name

    class _Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(TEST_HTML.encode())
        def log_message(self, *_):
            pass

    srv = HTTPServer(("localhost", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever)
    t.daemon = True
    t.start()

    with ExtractionEngine(headless=True) as engine:
        state = engine.capture_url(f"http://localhost:{port}/")

        print(f"URL:               {state.url}")
        print(f"Title:             {state.title}")
        print(f"DOM nodes:         {len(state.dom_index)}")
        print(f"A11y nodes:        {len(state.accessibility_index)}")
        print(f"Interactive els:   {len(state.interactive_elements)}")
        print(f"DOM->A11y refs:    {len(state.dom_to_a11y)}")
        print(f"Forms:             {state.meta.form_count}")
        print(f"Links:             {state.meta.link_count}")
        print(f"Language:          {state.meta.language}")

        print("\nDOM Tree:")
        print(engine.print_dom_tree(state.dom_tree, max_depth=4))

        print("Accessibility Tree:")
        print(engine.print_a11y_tree(state.accessibility_tree, max_depth=4))

        print("Interactive Elements:")
        for el in state.interactive_elements:
            print(f"  [{el.role}] {el.name or el.tag_name!r} -> {el.actions} | enabled={el.is_enabled}")

        state2 = engine.capture_url(f"http://localhost:{port}/")
        if state2.diff:
            print(f"\nDiff (same page recapture):")
            print(f"  Added:    {len(state2.diff.added_nodes)}")
            print(f"  Removed:  {len(state2.diff.removed_nodes)}")
            print(f"  Modified: {len(state2.diff.modified_nodes)}")
            print(f"  Significant change: {state2.diff.significant_change}")

    srv.shutdown()
    os.unlink(tmp_path)
    print("\nDone.")
