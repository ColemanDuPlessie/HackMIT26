import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Camera offset from the pelvis: in front of the subject (+x) and slightly to their left (+y),
// so the 3D view roughly matches a camera facing the person.
const CAMERA_OFFSET = new THREE.Vector3(3.0, 1.0, 0.6);

/**
 * three.js view of the MuJoCo humanoid. MuJoCo is z-up; geoms come from the server as
 * {type, size, rgba} and poses as flat [x, y, z, qw, qx, qy, qz] per geom.
 */
export class HumanoidView {
  constructor(stage, geoms, initialRoot = [0, 0, 0.93]) {
    this.stage = stage;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    stage.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e12);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(...initialRoot);
    camera.position.copy(controls.target).add(CAMERA_OFFSET);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x303640, 1.4);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 1.5, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3 });
    scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: 0x1a2028, roughness: 0.95 }),
    );
    ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(20, 40, 0x3a4450, 0x262d36);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; MuJoCo floor is XY
    grid.position.z = 0.001;
    scene.add(grid);

    // Humanoid and targets live in a group so the view can be mirrored (scale y = -1).
    this.body = new THREE.Group();
    scene.add(this.body);
    this.meshes = geoms.map((g) => {
      let geometry;
      if (g.type === 'capsule') {
        geometry = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 6, 16).rotateX(Math.PI / 2); // Y -> Z axis
      } else if (g.type === 'box') {
        geometry = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
      } else {
        geometry = new THREE.SphereGeometry(g.size[0], 24, 16);
      }
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
        roughness: 0.55,
        transparent: true,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      this.body.add(mesh);
      return mesh;
    });

    this.targetMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.022, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0x3ecf8e }),
      33,
    );
    this.targetMesh.count = 0;
    this.body.add(this.targetMesh);
    this.dummy = new THREE.Object3D();

    Object.assign(this, { renderer, scene, camera, controls });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(stage);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  /** Update geom transforms; optionally move the camera with the pelvis. */
  setPose(pose, root, follow = true) {
    this.meshes.forEach((mesh, i) => {
      const o = i * 7;
      mesh.position.set(pose[o], pose[o + 1], pose[o + 2]);
      mesh.quaternion.set(pose[o + 4], pose[o + 5], pose[o + 6], pose[o + 3]); // MuJoCo wxyz -> three xyzw
    });
    if (follow && root) {
      const worldRoot = this.body.localToWorld(new THREE.Vector3(...root));
      const delta = worldRoot.sub(this.controls.target);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
  }

  /** Show (33, 3) target points (null entries skipped), or hide them with null. */
  setTargets(targets) {
    let n = 0;
    if (targets) {
      for (const p of targets) {
        if (p[0] === null) continue;
        this.dummy.position.set(p[0], p[1], p[2]);
        this.dummy.updateMatrix();
        this.targetMesh.setMatrixAt(n++, this.dummy.matrix);
      }
    }
    this.targetMesh.count = n;
    this.targetMesh.instanceMatrix.needsUpdate = true;
  }

  /** Mirror the humanoid left-right, like a selfie camera. */
  setMirrored(mirrored) {
    this.body.scale.y = mirrored ? -1 : 1;
  }

  /** Fade the body, e.g. while tracking is lost. */
  setOpacity(opacity) {
    for (const mesh of this.meshes) mesh.material.opacity = opacity;
  }

  resize() {
    const { width, height } = this.stage.getBoundingClientRect();
    if (width > 0 && height > 0) {
      this.renderer.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    this.scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
