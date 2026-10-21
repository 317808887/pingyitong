"""Geometric RTL words, logical OCR characters, and top-to-bottom text output."""
import unittest

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import TextBlock
from pingyitong.components.render.render_item import RenderItem
from pingyitong.services.reading_order import order_for_reading, reading_text
from pingyitong.services.text_preprocess import TextPreprocessor, _order_line_blocks


def block(text, x, y=0, width=60):
    return TextBlock(text=text, box=Rect(x,y,width,30), confidence=.99)

class TestRtlReadingOrder(unittest.TestCase):
    def test_real_probe_geometric_blocks_restore_logical_word_order(self):
        blocks = [block("شكرا لكم",0), block("Order 12345",70), block("مرحبا بكم",140)]
        self.assertEqual([b.text for b in _order_line_blocks(blocks)], ["مرحبا بكم","Order 12345","شكرا لكم"])

    def test_latin_and_numeric_subruns_keep_geometric_order(self):
        blocks = [block("شكرا لكم",0), block("Order",70), block("12345",140), block("مرحبا بكم",210)]
        self.assertEqual([b.text for b in _order_line_blocks(blocks)], ["مرحبا بكم","Order","12345","شكرا لكم"])

    def test_single_block_never_reverses_decoded_characters(self):
        line = "رقم الطلب ١٢٣٤٥ متاح الآن"
        self.assertEqual(_order_line_blocks([block(line,0)])[0].text, line)

    def test_latin_majority_line_with_arabic_word_keeps_ltr(self):
        blocks = [block("Welcome to the application",0), block("مرحبا",70)]
        self.assertEqual(_order_line_blocks(blocks), blocks)

    def test_merge_and_assemble_preserve_top_to_bottom_and_rtl_words(self):
        pre = TextPreprocessor(filter_noise_enabled=False, same_line_gap_ratio=2)
        blocks = [block("مرحبا بكم",140),block("شكرا لكم",0),block("Order 12345",70)]
        result = pre._merge_group(blocks)
        self.assertEqual(result.text,"مرحبا بكم Order 12345 شكرا لكم")
        for b in blocks:
            b.line_id=0
        second=block("السطر الثاني",0,50)
        second.line_id=1
        self.assertEqual(pre._assemble_text([second]+blocks), "مرحبا بكم Order 12345 شكرا لكم السطر الثاني")

    def test_separate_widely_spaced_rtl_paragraphs_order_right_to_left(self):
        blocks = [block("شكرا لكم",961,width=177),block("Order 12345",1242,width=250),block("مرحبا بكم",1593,width=182)]
        self.assertEqual(reading_text(blocks), "مرحبا بكم\nOrder 12345\nشكرا لكم")

    def test_english_translations_keep_original_arabic_layout_direction(self):
        items = [RenderItem(Rect(961,25,177,81),"Thank you",source_text="شكرا لكم"),
                 RenderItem(Rect(1242,36,250,54),"Order 12345",source_text="Order 12345"),
                 RenderItem(Rect(1593,32,182,69),"Welcome",source_text="مرحبا بكم")]
        self.assertEqual(reading_text(items),"Welcome\nOrder 12345\nThank you")

    def test_arabic_translation_of_english_layout_does_not_reverse_source(self):
        items = [RenderItem(Rect(0,0,60,30),"مرحبا",source_text="Welcome"),
                 RenderItem(Rect(150,0,60,30),"شكرا",source_text="Thank you")]
        self.assertEqual(order_for_reading(items),items)

    def test_rtl_lines_still_top_to_bottom(self):
        items=[block("السطر الثاني",0,70,width=150),block("السطر الأول",0,0,width=150)]
        self.assertEqual(reading_text(items),"السطر الأول\nالسطر الثاني")

    def test_two_rtl_columns_each_remain_top_to_bottom(self):
        items=[block("شكرا لكم",0,0),block("السطر الثاني",0,45),
               block("مرحبا بكم",400,0),block("مرحبا مرة أخرى",400,45)]
        self.assertEqual(reading_text(items),"مرحبا بكم\nمرحبا مرة أخرى\nشكرا لكم\nالسطر الثاني")
