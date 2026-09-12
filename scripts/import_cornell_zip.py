"""Keep only publicly labelled Cornell photographs from an official Kaggle ZIP."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageOps


def import_archive(archive, output, delete_archive=False):
    archive, output = Path(archive).expanduser().resolve(), Path(output).expanduser().resolve()
    fields = ['healthy', 'multiple_diseases', 'rust', 'scab']
    archive_bytes = archive.stat().st_size
    with archive.open('rb') as handle:
        archive_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    with ZipFile(archive) as zipped:
        entries = zipped.infolist()
        tables = [entry for entry in entries if PurePosixPath(entry.filename).name == 'train.csv']
        if len(tables) != 1:
            raise ValueError('The official archive must contain exactly one train.csv.')
        csv_tables = []
        for entry in entries:
            if not entry.is_dir() and PurePosixPath(entry.filename).suffix.lower() == '.csv':
                table = csv.reader(io.StringIO(zipped.read(entry).decode('utf-8-sig')))
                csv_tables.append({'path': entry.filename, 'columns': next(table, []),
                                   'data_rows': sum(1 for row in table if row)})
        table_bytes = zipped.read(tables[0])
        reader = csv.DictReader(io.StringIO(table_bytes.decode('utf-8-sig')))
        if not {'image_id', *fields}.issubset(reader.fieldnames or []):
            raise ValueError('train.csv does not contain the expected publicly labelled columns.')
        rows = list(reader)
        ids, counts = set(), dict.fromkeys(fields, 0)
        for row in rows:
            identifier = row['image_id']
            values = [float(row[key]) for key in fields]
            if not re.fullmatch(r'Train_\d+', identifier) or identifier in ids:
                raise ValueError(f'Unexpected/repeated training image ID: {identifier!r}')
            if any(value not in (0, 1) for value in values) or sum(values) != 1:
                raise ValueError(f'Invalid one-hot label for {identifier}.')
            ids.add(identifier)
            counts[fields[values.index(1)]] += 1
        if not ids:
            raise ValueError('No labelled training rows were found.')
        images = {}
        image_entries = [entry for entry in entries if not entry.is_dir()
                         and PurePosixPath(entry.filename).suffix.lower()
                         in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}]
        ignored_test_images = sum(bool(re.fullmatch(r'Test_\d+', PurePosixPath(entry.filename).stem))
                                  for entry in image_entries)
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if name.suffix.lower() in {'.jpg', '.jpeg'} and name.stem in ids:
                if name.stem in images:
                    raise ValueError(f'Repeated archive image: {name.stem}')
                images[name.stem] = entry
        if set(images) != ids:
            raise ValueError(f'Missing {len(ids - set(images))} labelled images in archive.')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'images').mkdir(exist_ok=True)

        def preserve_file(path, body):
            digest = hashlib.sha256(body).digest()
            if path.exists():
                if hashlib.sha256(path.read_bytes()).digest() != digest:
                    raise ValueError(f'Existing file differs; refusing to replace {path}')
            else:
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(body)
                temporary.replace(path)
            if hashlib.sha256(path.read_bytes()).digest() != digest:
                raise OSError(f'Saved-file hash mismatch: {path}')

        receipt_images = []
        for index, identifier in enumerate(sorted(ids), 1):
            body = zipped.read(images[identifier])  # ZIP checks this member's CRC.
            with Image.open(io.BytesIO(body)) as photo:
                ImageOps.exif_transpose(photo).convert('RGB').load()
            relative = f'images/{identifier}.jpg'
            preserve_file(output / relative, body)
            receipt_images.append({'path': relative, 'bytes': len(body),
                                   'sha256': hashlib.sha256(body).hexdigest()})
            if index % 200 == 0:
                print(f'Verified/imported {index}/{len(ids)} labelled photos', flush=True)
        preserve_file(output / 'train.csv', table_bytes)
    receipt = {'source': 'https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data',
               'license': {'name': 'Apache 2.0, subject to competition rules',
                           'url': 'https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules',
                           'note': 'Competition rules take precedence; redistribution is restricted to people '
                                   'who accepted the rules. Keep this personal experiment\'s data private.'},
               'archive_sha256': archive_hash, 'archive_bytes': archive_bytes,
               'archive_image_entries': len(image_entries), 'csv_tables': csv_tables,
               'public_labeled_rows': len(rows), 'retained_images': len(ids), 'class_counts': counts,
               'retained_image_bytes': sum(entry['bytes'] for entry in receipt_images),
               'ignored_test_images': ignored_test_images,
               'ignored_other_images': len(image_entries) - len(ids) - ignored_test_images,
               'retained_metadata': ['train.csv'],
               'selection': 'Only images named in public train.csv; no test images or submission templates extracted.',
               'files': receipt_images, 'archive_deleted_after_verification': False}
    receipt_path = output / 'download_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if delete_archive:
        archive.unlink()
        receipt['archive_deleted_after_verification'] = True
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--delete-archive', action='store_true', help='Delete only this ZIP after selected images verify')
    args = parser.parse_args()
    import_archive(args.archive, args.output, args.delete_archive)
