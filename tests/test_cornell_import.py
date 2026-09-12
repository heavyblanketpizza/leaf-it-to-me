"""Verify selective Cornell import and release counts with a tiny local ZIP."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image, UnidentifiedImageError

from scripts.import_cornell_zip import import_archive


class CornellArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / 'download.zip'
        self.output = self.root / 'cornell'
        photo = io.BytesIO()
        Image.new('RGB', (8, 12), (30, 100, 40)).save(photo, format='JPEG')
        self.photo = photo.getvalue()

    def make_archive(self, train_image=None):
        with ZipFile(self.archive, 'w') as zipped:
            zipped.writestr('train.csv', 'image_id,healthy,multiple_diseases,rust,scab\nTrain_0,0,1,0,0\n')
            zipped.writestr('test.csv', 'image_id\nTest_0\nTest_1\n')
            zipped.writestr('sample_submission.csv',
                            'image_id,healthy,multiple_diseases,rust,scab\nTest_0,1,0,0,0\nTest_1,1,0,0,0\n')
            zipped.writestr('images/Train_0.jpg', self.photo if train_image is None else train_image)
            zipped.writestr('images/Test_0.jpg', self.photo)
            zipped.writestr('images/Test_1.JPG', self.photo)
            zipped.writestr('preview.png', b'not extracted or decoded')

    def test_inventory_selection_original_bytes_and_successful_cleanup(self):
        self.make_archive()
        archive_bytes = self.archive.read_bytes()
        with redirect_stdout(io.StringIO()):
            receipt = import_archive(self.archive, self.output, delete_archive=True)
        self.assertFalse(self.archive.exists())
        self.assertTrue(receipt['archive_deleted_after_verification'])
        self.assertEqual(receipt['archive_bytes'], len(archive_bytes))
        self.assertEqual(receipt['archive_sha256'], hashlib.sha256(archive_bytes).hexdigest())
        self.assertEqual(receipt['archive_image_entries'], 4)
        self.assertEqual(receipt['public_labeled_rows'], 1)
        self.assertEqual(receipt['retained_images'], 1)
        self.assertEqual(receipt['ignored_test_images'], 2)
        self.assertEqual(receipt['ignored_other_images'], 1)
        self.assertEqual(receipt['class_counts']['multiple_diseases'], 1)
        self.assertEqual(receipt['retained_image_bytes'], len(self.photo))
        tables = {table['path']: table for table in receipt['csv_tables']}
        self.assertEqual(tables['test.csv'], {'path': 'test.csv', 'columns': ['image_id'], 'data_rows': 2})
        self.assertEqual(tables['train.csv']['columns'],
                         ['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])
        self.assertEqual(tables['train.csv']['data_rows'], 1)
        self.assertEqual((self.output / 'images/Train_0.jpg').read_bytes(), self.photo)
        self.assertEqual({path.relative_to(self.output).as_posix() for path in self.output.rglob('*')
                          if path.is_file()}, {'train.csv', 'images/Train_0.jpg', 'download_receipt.json'})
        self.assertEqual(json.loads((self.output / 'download_receipt.json').read_text()), receipt)

    def test_unreadable_training_image_keeps_archive(self):
        self.make_archive(train_image=b'broken image')
        with self.assertRaises(UnidentifiedImageError), redirect_stdout(io.StringIO()):
            import_archive(self.archive, self.output, delete_archive=True)
        self.assertTrue(self.archive.exists())
        self.assertFalse((self.output / 'download_receipt.json').exists())

    def test_failed_cleanup_does_not_claim_archive_was_deleted(self):
        self.make_archive()
        with patch.object(Path, 'unlink', side_effect=PermissionError('read-only archive')):
            with self.assertRaises(PermissionError), redirect_stdout(io.StringIO()):
                import_archive(self.archive, self.output, delete_archive=True)
        self.assertTrue(self.archive.exists())
        receipt = json.loads((self.output / 'download_receipt.json').read_text())
        self.assertFalse(receipt['archive_deleted_after_verification'])


if __name__ == '__main__':
    unittest.main()
