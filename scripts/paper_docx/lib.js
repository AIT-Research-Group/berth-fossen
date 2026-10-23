// Shared helpers for the results document: styled tables, figures and bilingual text blocks.
const fs = require('fs');
const path = require('path');
const {
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType, BorderStyle, ShadingType,
  ImageRun, HeadingLevel, VerticalAlign,
} = require('docx');

const FONT = 'Times New Roman';
const TEXT_WIDTH = 9026;  // A4 with 1-inch margins, in DXA
const thin = {style: BorderStyle.SINGLE, size: 4, color: '000000'};
const none = {style: BorderStyle.NONE, size: 0, color: 'FFFFFF'};

function run(text, options = {}) {
  return new TextRun({text, font: FONT, size: options.size || 22, bold: options.bold, italics: options.italics,
                      color: options.color, superScript: options.superScript, subScript: options.subScript});
}

// Text with simple markup: ~x~ subscript, ^x^ superscript, e.g. "H~m0,R~" (paths keep their underscores).
function rich(text, options = {}) {
  const parts = [];
  const pattern = /(~[^~]+~|\^[^^]+\^)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push(run(text.slice(last, match.index), options));
    const body = match[0].slice(1, -1);
    parts.push(run(body, {...options, subScript: match[0][0] === '~', superScript: match[0][0] === '^'}));
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(run(text.slice(last), options));
  return parts;
}

function para(text, options = {}) {
  return new Paragraph({children: rich(text, options), alignment: options.align || AlignmentType.JUSTIFIED,
                        spacing: {after: options.after ?? 120, before: options.before ?? 0, line: 276},
                        keepNext: options.keepNext});
}

function heading(text, level) {
  return new Paragraph({heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
                        children: [run(text, {bold: true, size: level === 1 ? 28 : 24})],
                        spacing: {before: level === 1 ? 360 : 240, after: 120}, keepNext: true});
}

// Vietnamese commentary, visually distinct from the English paper material.
function note(label, text) {
  return new Paragraph({
    children: [run(`${label}: `, {bold: true, size: 21, color: '1F4E79'}), ...rich(text, {size: 21, color: '1F4E79'})],
    alignment: AlignmentType.JUSTIFIED, spacing: {after: 160, line: 276},
    border: {left: {style: BorderStyle.SINGLE, size: 12, color: '1F4E79', space: 6}}, indent: {left: 120},
  });
}

function caption(label, text) {
  return new Paragraph({children: [run(`${label} `, {bold: true, size: 20}), ...rich(text, {size: 20})],
                        alignment: AlignmentType.JUSTIFIED, spacing: {before: 80, after: 160}, keepNext: false});
}

function tableCaption(label, text) {
  return new Paragraph({children: [run(`${label} `, {bold: true, size: 20}), ...rich(text, {size: 20})],
                        spacing: {before: 200, after: 80}, keepNext: true});
}

// Booktabs-style table: rules above/below the header and at the bottom only.
function table(header, rows, widths, options = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map(w => Math.round(w * TEXT_WIDTH / total));
  scaled[scaled.length - 1] += TEXT_WIDTH - scaled.reduce((a, b) => a + b, 0);
  const align = options.align || header.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT));
  const cell = (text, column, rowIndex, isHeader, isLast, shade) => new TableCell({
    width: {size: scaled[column], type: WidthType.DXA}, verticalAlign: VerticalAlign.CENTER,
    margins: {top: 40, bottom: 40, left: 80, right: 80},
    shading: shade ? {type: ShadingType.CLEAR, color: 'auto', fill: shade} : undefined,
    borders: {top: isHeader ? thin : none, bottom: isHeader || isLast ? thin : none, left: none, right: none},
    children: [new Paragraph({children: rich(String(text), {size: 18, bold: isHeader}),
                              alignment: isHeader ? AlignmentType.CENTER : align[column]})],
  });
  const body = rows.map((row, r) => new TableRow({
    cantSplit: true,
    children: row.map((text, c) => cell(text, c, r, false, r === rows.length - 1,
                                        options.shade && options.shade(r) ? 'F2F2F2' : undefined)),
  }));
  return new Table({width: {size: TEXT_WIDTH, type: WidthType.DXA}, columnWidths: scaled,
                    rows: [new TableRow({tableHeader: true, children: header.map((h, c) => cell(h, c, -1, true, false))}),
                           ...body]});
}

// PNG size from its IHDR chunk, to keep the aspect ratio.
function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return {width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20), buffer};
}

function figure(file, widthInches = 6.27) {
  const {width, height, buffer} = pngSize(file);
  const w = Math.round(widthInches * 96);
  return new Paragraph({alignment: AlignmentType.CENTER, spacing: {before: 120, after: 0}, keepNext: true,
                        children: [new ImageRun({type: 'png', data: buffer,
                                                 transformation: {width: w, height: Math.round(w * height / width)},
                                                 altText: {title: path.basename(file), description: path.basename(file),
                                                           name: path.basename(file)}})]});
}

const fmt = (value, digits = 3) => (value === '' || value === null || value === undefined || Number.isNaN(+value))
  ? '—' : (+value).toFixed(digits);

module.exports = {FONT, TEXT_WIDTH, run, rich, para, heading, note, caption, tableCaption, table, figure, fmt};
