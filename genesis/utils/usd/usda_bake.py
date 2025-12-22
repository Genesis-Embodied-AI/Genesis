import argparse
import asyncio
import logging
import os
import time


def omni_bootstrap(device=0, log_level="warning"):
    import omni.kit_app

    app = omni.kit_app.KitApp()
    kit_dir = os.path.dirname(os.path.abspath(os.path.realpath(omni.kit_app.__file__)))
    kit_path = os.path.join(kit_dir, "apps", "omni.app.empty.kit")
    app_args = [
        kit_path,
        "--/app/window/hideUi=True",
        f"--/app/tokens/exe-path={kit_dir}",
        "--/app/enableStdoutOutput=False",  # Disable print outs (print_and_log) on extension startup information
        "--/app/runLoops/present/rateLimitFrequency=60",
        "--/app/vulkan=True",
        "--/app/asyncRendering=False",
        "--/app/python/interceptSysStdOutput=False",
        "--/app/python/logSysStdOutput=False",
        "--/app/settings/fabricDefaultStageFrameHistoryCount=3",
        f"--/omni/log/level={log_level}",
        "--/log/file=",  # Empty string means no log file
        f"--/log/level={log_level}",
        f"--/renderer/activeGpu={device}",
        "--/renderer/enabled=rtx",
        "--/renderer/active=rtx",
        "--/renderer/multiGpu/enabled=False",  # Avoids unnecessary GPU context initialization
        "--no-window",
        "--portable",
        "--enable",
        "omni.usd",
        "--enable",
        "omni.kit.material.library",
        "--enable",
        "omni.kit.viewport.utility",
        "--enable",
        "omni.kit.viewport.rtx",
        "--enable",
        "omni.kit.usd.collect",
        "--enable",
        "omni.replicator.core",
        "--enable",
        "omni.mdl.distill_and_bake",
    ]
    app.startup(app_args)
    app.update()  # important
    return app


def bake_usd_material(input_file, output_dir, usd_material_paths, device=0, log_level="error"):
    logs = []

    # bootstrap
    start_time = time.time()
    app = omni_bootstrap(device, log_level)
    logs.append(f"\tBootstrap: {time.time() - start_time}, App status: {app.is_running()}.")

    import omni.usd
    import omni.mdl.distill_and_bake
    import omni.replicator.core
    import omni.kit.usd.collect

    # open stage
    start_time = time.time()
    omni.usd.get_context().open_stage(input_file)
    logs.append(f"\tOpen stage: {time.time() - start_time}s.")

    # create render product
    start_time = time.time()
    stage = omni.usd.get_context().get_stage()
    render_prod_path = omni.replicator.core.create.render_product("/OmniverseKit_Persp", resolution=(600, 600))
    app.update()  # important
    logs.append(f"\tCreate render product: {time.time() - start_time}s, {render_prod_path}.")

    # distill the material
    start_time = time.time()

    for usd_material_path in usd_material_paths:
        material_prim = stage.GetPrimAtPath(usd_material_path)
        distiller = omni.mdl.distill_and_bake.MdlDistillAndBake(material_prim, ouput_folder=output_dir)
        distiller.distill()
    logs.append(f"\tDistill: {time.time() - start_time}s, {material_prim}.")

    # export usd
    start_time = time.time()
    collector = omni.kit.usd.collect.Collector(stage.GetRootLayer().identifier, output_dir)
    task = asyncio.ensure_future(collector.collect())
    while not task.done():
        app.update()  # Otherwise it will be blocked by omni.kit.app.get_app().next_update_async()
    success, baked_file = task.result()

    logs.append(f"\tExport: {time.time() - start_time}s, {baked_file}.")
    print("Distill USD material:\n" + "\n".join(logs))

    # close omniverse app
    app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--usd_material_paths", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="warning")
    args = parser.parse_args()

    log_level = logging.getLevelName(
        min(max(logging.getLevelName(args.log_level.upper()), logging.INFO), logging.ERROR)
    ).lower()
    bake_usd_material(args.input_file, args.output_dir, args.usd_material_paths, args.device, log_level)
