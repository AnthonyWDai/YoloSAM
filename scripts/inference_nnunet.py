import os
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict

import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.yolo import YOLOModel
from models.sam import SAMModel
from utils.config import YOLOConfig, SAMFinetuneConfig, YoloSAMInferenceConfig
from utils.prompt import BoxPromptGenerator

# nnU-Net imports
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json


class YoloSAMInference:
    def __init__(
        self,
        config: YoloSAMInferenceConfig,
        dataset_json_path: str,
        plans_json_path: str,
        model_training_output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        YOLOSAM inference using nnU-Net-style preprocessing/export.

        Parameters
        ----------
        config : YoloSAMInferenceConfig
            YOLOSAM config.
        dataset_json_path : str
            Path to nnU-Net dataset.json.
        plans_json_path : str
            Path to nnU-Net plans.json.
        model_training_output_dir : Optional[str]
            Optional path to nnU-Net trained model folder if you want to reuse metadata structure.
        device : Optional[str]
            Override device.
        """
        self.config = config
        self.device = device or config.device

        # ---------------------------
        # nnU-Net metadata
        # ---------------------------
        self.dataset_json = load_json(dataset_json_path)
        self.plans = load_json(plans_json_path)
        self.plans_manager = PlansManager(self.plans)

        if hasattr(config, "nnunet_configuration") and config.nnunet_configuration is not None:
            self.configuration_name = config.nnunet_configuration
        else:
            self.configuration_name = "3d_fullres"

        self.configuration_manager = self.plans_manager.get_configuration(self.configuration_name)

        # # reader/writer from dataset.json
        # io_class = determine_reader_writer_from_dataset_json(
        #     self.dataset_json,
        #     self.plans_manager
        # )
        # self.reader_writer = io_class()

        # preprocessor
        self.preprocessor = self.configuration_manager.preprocessor_class(verbose=False)

        # ---------------------------
        # Initialize YOLO
        # ---------------------------
        self.yolo_config = YOLOConfig(
            checkpoint_path=config.yolo_checkpoint_path,
            device=self.device,
            conf_threshold=config.yolo_conf_threshold,
            iou_threshold=config.yolo_iou_threshold,
            max_detections=config.yolo_max_detections,
            dataset_path=None,
            val_dataset_path=None
        )
        self.yolo_model = YOLOModel(self.yolo_config)

        # ---------------------------
        # Initialize SAM
        # ---------------------------
        self.sam_config = SAMFinetuneConfig(
            sam_path=None,
            checkpoint_path=config.sam_checkpoint_path,
            model_type="vit_b",
            device=self.device
        )
        self.sam_model = SAMModel(self.sam_config)

        # prompt generator
        self.box_prompt_generator = BoxPromptGenerator(
            enable_direction_aug=False,
            enable_size_aug=False
        )

        print(f"YoloSAM nnU-Net-style inference initialized on {self.device}")

    def _normalize_to_uint8(self, x: np.ndarray) -> np.ndarray:
        """Normalize one 2D slice to uint8 [0,255]."""
        x = x.astype(np.float32)
        mn, mx = np.percentile(x, 0.5), np.percentile(x, 99.5)
        if mx <= mn:
            return np.zeros_like(x, dtype=np.uint8)
        x = np.clip((x - mn) / (mx - mn), 0, 1)
        return (x * 255).astype(np.uint8)

    def _slice_to_rgb(self, slice_2ch: np.ndarray) -> np.ndarray:
        """
        Convert a 2-channel slice [2, H, W] into RGB [H, W, 3]:
        ch0 = modality 0
        ch1 = modality 1
        ch2 = average(ch0, ch1)
        """
        assert slice_2ch.ndim == 3 and slice_2ch.shape[0] == 2, \
            f"Expected slice shape [2, H, W], got {slice_2ch.shape}"

        ch0 = self._normalize_to_uint8(slice_2ch[0])
        ch1 = self._normalize_to_uint8(slice_2ch[1])
        ch2 = ((ch0.astype(np.float32) + ch1.astype(np.float32)) / 2.0).astype(np.uint8)

        rgb = np.stack([ch0, ch1, ch2], axis=-1)
        return rgb

    def preprocess_case_with_nnunet(
        self,
        input_files: List[str],
        seg_prev_stage: Optional[str] = None
    ):
        data, seg, data_properties = self.preprocessor.run_case(
            input_files,
            seg_prev_stage,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json
        )
        return data, data_properties

    def detect_objects(self, image_rgb: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO model on a single RGB slice."""
        results = self.yolo_model.predict(image_rgb, verbose=False)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    conf = float(box.conf[0].detach().cpu().numpy())
                    cls = int(box.cls[0].detach().cpu().numpy())

                    detections.append({
                        'bbox': xyxy,
                        'confidence': conf,
                        'class': cls,
                        'class_name': self.yolo_config.class_names[cls]
                        if cls < len(self.yolo_config.class_names) else f'class_{cls}'
                    })
        return detections

    def segment_with_sam(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """Run SAM for one 2D RGB slice."""
        results = []

        h_orig, w_orig = image_rgb.shape[:2]

        sam_image = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        sam_image_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).float() / 255.0
        sam_image_tensor = sam_image_tensor.unsqueeze(0).to(self.device)

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            x1_sam = int(x1 * 1024 / w_orig)
            y1_sam = int(y1 * 1024 / h_orig)
            x2_sam = int(x2 * 1024 / w_orig)
            y2_sam = int(y2 * 1024 / h_orig)

            sam_bbox = torch.tensor([x1_sam, y1_sam, x2_sam, y2_sam], dtype=torch.float32).to(self.device)

            mask, iou_pred = self.sam_model.forward_one_image(
                image=sam_image_tensor,
                bounding_box=sam_bbox,
                is_train=False
            )

            mask = torch.sigmoid(mask)
            mask_binary = (mask > 0.5).float()
            mask_np = mask_binary[0, 0].detach().cpu().numpy()

            mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            results.append({
                'bbox': bbox,
                'mask': mask_resized,
                'confidence': detection['confidence'],
                'class': detection['class'],
                'class_name': detection['class_name'],
                'iou_prediction': iou_pred[0].detach().cpu().item()
            })

        return results

    def predict_preprocessed_volume(
        self,
        data: np.ndarray,
        merge_mode: str = "union"
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predict a preprocessed nnU-Net volume slice by slice.

        Parameters
        ----------
        data : np.ndarray
            Preprocessed array, expected shape [C, X, Y, Z] with C >= 2
        merge_mode : str
            How to merge multiple instance masks on each slice.
            'union' => binary foreground union

        Returns
        -------
        pred_seg : np.ndarray
            3D segmentation in preprocessed space, shape [X, Y, Z]
        slice_results : List[Dict]
            Per-slice outputs for debugging/inspection
        """
        assert data.ndim == 4, f"Expected preprocessed data [C, X, Y, Z], got {data.shape}"
        assert data.shape[0] >= 2, f"Expected at least 2 modalities/channels, got {data.shape[0]}"

        c, x, y, z = data.shape
        pred_seg = np.zeros((x, y, z), dtype=np.uint8)
        slice_results = []

        for iz in range(z):
            slice_2ch = data[:2, :, :, iz]
            image_rgb = self._slice_to_rgb(slice_2ch)

            detections = self.detect_objects(image_rgb)

            if len(detections) == 0:
                slice_results.append({
                    "slice_index": iz,
                    "detections": [],
                    "message": "No objects detected"
                })
                continue

            segmentation_results = self.segment_with_sam(image_rgb, detections)

            slice_mask = np.zeros((x, y), dtype=np.uint8)

            for det in segmentation_results:
                mask = (det["mask"] > 0.5).astype(np.uint8)
                if merge_mode == "union":
                    slice_mask = np.maximum(slice_mask, mask)
                else:
                    slice_mask = np.maximum(slice_mask, mask)

            pred_seg[:, :, iz] = slice_mask

            slice_results.append({
                "slice_index": iz,
                "detections": segmentation_results,
                "message": f"Detected and segmented {len(segmentation_results)} objects"
            })

        return pred_seg, slice_results

    def export_prediction_nnunet_style(
        self,
        pred_seg_preprocessed: np.ndarray,
        data_properties: dict,
        output_file_truncated: str,
        save_probabilities: bool = False
    ):
        """
        Export prediction back to original image geometry similar to nnU-Net.

        nnU-Net export function expects logits, not hard labels. Since YOLOSAM outputs a binary mask,
        we convert it into 2-class logits-like scores.
        """
        pred_seg_preprocessed = pred_seg_preprocessed.astype(np.float32)

        # Fake 2-class logits/probabilities-like tensor:
        # background = 1 - mask, foreground = mask
        bg = 1.0 - pred_seg_preprocessed
        fg = pred_seg_preprocessed
        logits = np.stack([bg, fg], axis=0).astype(np.float32)

        export_prediction_from_logits(
            logits,
            data_properties,
            self.configuration_manager,
            self.plans_manager,
            self.dataset_json,
            output_file_truncated,
            save_probabilities
        )

    def predict_case(
        self,
        input_files: List[str],
        output_file_truncated: Optional[str] = None,
        save_slices: bool = False,
        slices_output_dir: Optional[str] = None,
        save_probabilities: bool = False
    ) -> Dict:
        """
        Predict one 3D case from two-modality NIfTI input.

        Parameters
        ----------
        input_files : List[str]
            Two modality files:
            [case_XXX_0000.nii.gz, case_XXX_0001.nii.gz]
        output_file_truncated : Optional[str]
            Output path without file ending, same as nnU-Net convention.
        save_slices : bool
            Whether to save per-slice PNG masks.
        slices_output_dir : Optional[str]
            Folder for saving per-slice outputs.
        save_probabilities : bool
            Forwarded to nnU-Net exporter.

        Returns
        -------
        Dict with metadata and optional slice results.
        """
        data, data_properties = self.preprocess_case_with_nnunet(input_files)

        pred_seg_preprocessed, slice_results = self.predict_preprocessed_volume(data)

        if save_slices and slices_output_dir is not None:
            maybe_mkdir_p(slices_output_dir)
            for iz in range(pred_seg_preprocessed.shape[-1]):
                out_png = join(slices_output_dir, f"slice_{iz:04d}.png")
                cv2.imwrite(out_png, (pred_seg_preprocessed[:, :, iz] * 255).astype(np.uint8))

        if output_file_truncated is not None:
            self.export_prediction_nnunet_style(
                pred_seg_preprocessed,
                data_properties,
                output_file_truncated,
                save_probabilities=save_probabilities
            )

        return {
            "input_files": input_files,
            "preprocessed_shape": tuple(data.shape),
            "prediction_shape_preprocessed": tuple(pred_seg_preprocessed.shape),
            "num_slices": pred_seg_preprocessed.shape[-1],
            "slice_results": slice_results,
            "output_file_truncated": output_file_truncated
        }

    def predict_from_files(
        self,
        list_of_cases: List[List[str]],
        output_folder: str,
        save_probabilities: bool = False,
        overwrite: bool = True,
        save_slices: bool = False
    ) -> List[Dict]:
        """
        Batch prediction similar in spirit to nnU-Net predict_from_files, but sequential and YOLOSAM-based.

        list_of_cases example:
        [
            ["/path/case_001_0000.nii.gz", "/path/case_001_0001.nii.gz"],
            ["/path/case_002_0000.nii.gz", "/path/case_002_0001.nii.gz"]
        ]
        """
        maybe_mkdir_p(output_folder)

        save_json(self.dataset_json, join(output_folder, "dataset.json"), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, "plans.json"), sort_keys=False)

        results_all = []

        for case_files in list_of_cases:
            assert len(case_files) == 2, f"Expected 2 modality files, got {len(case_files)}: {case_files}"

            case_id = os.path.basename(case_files[0]).replace("_0000.nii.gz", "")
            output_file_truncated = join(output_folder, case_id)

            out_seg = output_file_truncated + self.dataset_json["file_ending"]
            if (not overwrite) and os.path.isfile(out_seg):
                print(f"Skipping existing case: {case_id}")
                continue

            print(f"Predicting case: {case_id}")

            slices_output_dir = join(output_folder, f"{case_id}_slices") if save_slices else None

            result = self.predict_case(
                input_files=case_files,
                output_file_truncated=output_file_truncated,
                save_slices=save_slices,
                slices_output_dir=slices_output_dir,
                save_probabilities=save_probabilities
            )
            results_all.append(result)

        return results_all


def build_case_list_from_folder(input_folder: str, file_ending: str = ".nii.gz") -> List[List[str]]:
    """
    Build case list for two-modality nnU-Net input:
    case_001_0000.nii.gz
    case_001_0001.nii.gz
    """
    input_folder = Path(input_folder)

    mod0_files = sorted(input_folder.glob(f"*_0000{file_ending}"))
    cases = []

    for f0 in mod0_files:
        case_id = f0.name.replace(f"_0000{file_ending}", "")
        f1 = input_folder / f"{case_id}_0001{file_ending}"
        if not f1.exists():
            raise FileNotFoundError(f"Missing modality 1 for case {case_id}: {f1}")
        cases.append([str(f0), str(f1)])

    return cases


def main():
    parser = argparse.ArgumentParser(
        description="YOLOSAM inference with nnU-Net-style preprocessing/export for 3D two-modality NIfTI data."
    )

    parser.add_argument(
        "-i", type=str, required=True,
        help="Input folder containing cases as *_0000.nii.gz and *_0001.nii.gz"
    )
    parser.add_argument(
        "-o", type=str, required=True,
        help="Output folder for predicted nii.gz files"
    )
    parser.add_argument(
        "--dataset_json", type=str, required=True,
        help="Path to nnU-Net dataset.json"
    )
    parser.add_argument(
        "--plans_json", type=str, required=True,
        help="Path to nnU-Net plans.json"
    )
    parser.add_argument(
        "--yolo_checkpoint", type=str, required=True,
        help="Path to YOLO checkpoint"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True,
        help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", required=False,
        help="Device for inference: cuda, cpu, or mps"
    )
    parser.add_argument(
        "--save_slices", action="store_true",
        help="Save per-slice prediction masks as PNG"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing predictions"
    )
    parser.add_argument(
        "--save_probabilities", action="store_true",
        help="Save probabilities if supported by export"
    )
    parser.add_argument(
        "--yolo_conf", type=float, default=0.25,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--yolo_iou", type=float, default=0.45,
        help="YOLO IoU threshold"
    )
    parser.add_argument(
        "--yolo_max_det", type=int, default=100,
        help="YOLO max detections per slice"
    )
    parser.add_argument(
        "--nnunet_configuration", type=str, default="3d_fullres",
        help="nnU-Net configuration name from plans.json"
    )

    args = parser.parse_args()

    maybe_mkdir_p(args.o)

    config = YoloSAMInferenceConfig(
        yolo_checkpoint_path=args.yolo_checkpoint,
        sam_checkpoint_path=args.sam_checkpoint,
        device=args.device,
        yolo_conf_threshold=args.yolo_conf,
        yolo_iou_threshold=args.yolo_iou,
        yolo_max_detections=args.yolo_max_det
    )

    # if config class supports this attribute
    config.nnunet_configuration = args.nnunet_configuration

    inference_pipeline = YoloSAMInference(
        config=config,
        dataset_json_path=args.dataset_json,
        plans_json_path=args.plans_json
    )

    case_list = build_case_list_from_folder(args.i, file_ending=".nii.gz")

    results = inference_pipeline.predict_from_files(
        list_of_cases=case_list,
        output_folder=args.o,
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        save_slices=args.save_slices
    )

    print(f"Done. Predicted {len(results)} cases.")


if __name__ == "__main__":
    # python inference_nnunet.py \
    # -i /path/to/imagesTs \
    # -o /path/to/predictions \
    # --dataset_json /path/to/dataset.json \
    # --plans_json /path/to/plans.json \
    # --yolo_checkpoint runs/yolo_scar_detection2/weights/best.pt \
    # --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth \
    # --device cuda
    main()