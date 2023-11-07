import torch
from torch_geometric.data import Dataset, Data, extract_zip
from torch.utils.data.dataset import IterableDataset
import gdown
import os, shutil, tqdm
from typing import Iterator, List, Callable
from pyntcloud import PyntCloud
from more_itertools import chunked
import pickle
import glob
import warnings


class ShapeNetCompletion(Dataset, IterableDataset):
    id = '1hHIoAW97HUsc2A9F159xutd0ajar1mqi'
    name = 'ShapeNetCompletion'

    category_ids = {
        'Airplane': '02691156',
        'Cabinet': '02933112',
        'Car': '02958343',
        'Chair': '03001627',
        'Lamp': '03636649',
        'Sofa': '04256520',
        'Table': '04379243',
        'Watercraft': '04530566'
    }
    base_splits = ['train', 'val', 'test']

    def __init__(
            self, 
            root: str, 
            chunks: int = 16,
            categories: str | list[str] | None = None, 
            split: str = 'train', 
            transform: Callable | None=None, 
            pre_transform: Callable | None=None, 
            pre_filter: Callable | None=None, 
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        self.categories = categories
        self.cat_joined = '_'.join([cat[:3].lower() for cat in self.categories])
        self.chunks = chunks
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self) -> List[str]:
        return self.base_splits
        
    @property
    def processed_file_names(self) -> str | list[str] | tuple:
        return [
            os.path.join(f'{self.cat_joined}_{self.split}.pt.{part}') for part in range(self.chunks)
        ]
    
    def download(self):
        assert self.root
        outzip = gdown.download(id=self.id, output=self.root + '.zip', quiet=False, resume=True)
        extract_zip(outzip, self.root)
        os.unlink(outzip)
        shutil.rmtree(self.raw_dir)
        os.rename(os.path.join(self.root, self.name), self.raw_dir)
    
    def process(self) -> None:
        names = list(self._iter_names(self.split))
        for part, chunk in enumerate(tqdm.tqdm(chunked(names, len(names) // self.chunks))):
            data_list = self._load_data_for_chunk(chunk)
            with open(os.path.join(self.processed_dir, f'{self.cat_joined}_{self.split}.pt.{part}'), 'wb') as f:
                pickle.dump(data_list, f)

    def _process(self):
        chunk_file = os.path.join(self.processed_dir, f'chunks_{self.split}.pt')
        if os.path.exists(chunk_file):
            with open(chunk_file, 'rb') as f:
                if pickle.load(f) != self.chunks:
                    warnings.warn(
                        f"The `chunks` argument differs from the one used in "
                        f"the pre-processed version of this dataset. If you want to "
                        f"make use of another chunk size, make sure to "
                        f"delete '{self.processed_dir}' first")
                    
        super()._process()

        with open(chunk_file, 'wb') as f:
            pickle.dump(self.chunks, f)

    def __iter__(self) -> Iterator[Data]:
        for part in range(self.chunks):
            file = os.path.join(self.processed_dir, f'{self.cat_joined}_{self.split}.pt.{part}')
            with open(file, 'rb') as f:
                data_list = pickle.load(f)
            
            for data in data_list:
                yield data if self.transform is None else self.transform(data)
                

    def _iter_names(self, split: str):
        c_dir = os.path.join(self.raw_dir, split, 'complete')
        p_dir = os.path.join(self.raw_dir, split, 'partial')

        for cat_name in self.categories:
            cat_id = self.category_ids[cat_name]
            p_dir_cat = os.path.join(p_dir, cat_id)
            for name in os.listdir(p_dir_cat):
                c_name, p_dir_name = os.path.join(c_dir, cat_id, f'{name}.pcd'), os.path.join(p_dir_cat, name)
                p_names = glob.glob(os.path.join(p_dir_name, '*.pcd'))
                yield c_name, p_names

    def _load_data_for_chunk(self, chunk) -> list[Data]:
        data_list = []
        for (c_name, p_names) in chunk:
            c_point_cloud = torch.tensor(PyntCloud.from_file(c_name).points.values)
            p_point_clouds = (torch.tensor(PyntCloud.from_file(p_name).points.values) for p_name in p_names)
            data = (Data(pos=p, y=c_point_cloud) for p in p_point_clouds)
            if self.pre_filter is not None:
                data = (d for d in data if self.pre_filter(d))
            if self.pre_transform is not None:
                data = (self.pre_transform(d) for d in data)
            data_list.extend(data)
        return data_list
    
    def get(self, idx: int) -> Data:
        raise NotImplementedError("Use __iter__ instead")
    
    def len(self) -> int:
        raise NotImplementedError("Use __iter__ instead")
